#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

// ===================================================================================
// Helper for CUDA Error Handling
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// ===================================================================================
// Data Loading
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; 
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

// ===================================================================================
// Constant Memory
// ===================================================================================
__constant__ float c_conv1_w[150];
__constant__ float c_conv1_b[6];
__constant__ float c_conv2_w[2400];
__constant__ float c_conv2_b[16];

// ===================================================================================
// Kernels (V6: Max Parallelism)
// ===================================================================================

// Kernel 1: Fused Conv1 -> IF -> Pool1
// Block Dim: (12, 12, 6) -> 864 threads. One thread per output neuron.
// This eliminates the loop over channels C.
__global__ void k_conv1_max_parallel(
    const float* __restrict__ input,
    float* __restrict__ out_spikes,
    int N
) {
    // Shared Memory: 28x32 (Padded)
    __shared__ float s_img[28][32];
    
    int b = blockIdx.x;
    
    // Map Thread ID to Output Dimensions
    // blockDim.x = 12 (W), blockDim.y = 12 (H), blockDim.z = 6 (C)
    int tx = threadIdx.x; // 0..11 (W_out)
    int ty = threadIdx.y; // 0..11 (H_out)
    int tc = threadIdx.z; // 0..5  (Channel)
    
    // Linear ID for loading (0..863)
    int tid = tc * 144 + ty * 12 + tx;

    // 1. Cooperative Load (196 float4s needed)
    const float4* in_ptr = reinterpret_cast<const float4*>(input + b * 784);
    if (tid < 196) {
        float4 v = in_ptr[tid];
        int idx = tid * 4;
        s_img[idx/28][idx%28] = v.x; 
        s_img[(idx+1)/28][(idx+1)%28] = v.y;
        s_img[(idx+2)/28][(idx+2)%28] = v.z; 
        s_img[(idx+3)/28][(idx+3)%28] = v.w;
    }
    __syncthreads();

    // 2. Pre-compute Current (I)
    // Each thread computes ONE output value (covering 2x2 input window)
    float I[4]; // Current for the 4 sub-pixels in the 2x2 pooling window
    
    // Init with Bias
    float bias = c_conv1_b[tc];
    #pragma unroll
    for(int i=0; i<4; ++i) I[i] = bias;

    // Base position in Input (stride 2)
    int h_base = ty * 2;
    int w_base = tx * 2;

    // Pre-calculate weight offset for this channel
    // Weights: [6][5][5]
    int w_offset_base = tc * 25;

    // Iterate 2x2 Pooling Window
    #pragma unroll
    for (int py = 0; py < 2; ++py) {
        #pragma unroll
        for (int px = 0; px < 2; ++px) {
            int sub_idx = py * 2 + px;
            int h_in = h_base + py;
            int w_in = w_base + px;

            // Convolution 5x5
            #pragma unroll
            for (int kh = 0; kh < 5; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 5; ++kw) {
                    float val = s_img[h_in + kh][w_in + kw];
                    I[sub_idx] += val * c_conv1_w[w_offset_base + kh * 5 + kw];
                }
            }
        }
    }

    // 3. Time Evolution (T=8)
    float v[4]; // Voltage state
    #pragma unroll
    for(int i=0; i<4; ++i) v[i] = 0.0f;

    long long base_addr = (long long)b * 6912 + (tc * 144 + ty * 12 + tx);

    for (int t = 0; t < 8; ++t) {
        float max_spike = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float voltage = v[i] + I[i];
            float spike = (voltage >= 1.0f) ? 1.0f : 0.0f;
            
            if (spike > 0.5f) v[i] = 0.0f; 
            else v[i] = voltage;

            if (spike > max_spike) max_spike = spike;
        }
        
        // Output: [Batch, Time, Channel, Height, Width]
        // Address stride for Time is 864 (6*12*12)
        out_spikes[base_addr + t * 864] = max_spike;
    }
}

// Kernel 2: Fused Conv2 -> IF -> Pool2
// Block Dim: 256 threads (1D). Each thread maps to (c, y, x) output.
// Output: 16 ch * 4 h * 4 w = 256 elements. Perfect match.
__global__ void k_conv2_max_parallel(
    const float* __restrict__ in_spikes,
    float* __restrict__ out_spikes,
    int N
) {
    __shared__ float s_spikes[8][6][12][12]; // 27KB

    int b = blockIdx.x;
    int tid = threadIdx.x; // 0..255

    // 1. Cooperative Load
    // 6912 floats / 256 threads = 27 loads exactly
    const float* src = in_spikes + (long long)b * 6912;
    float* dst = &s_spikes[0][0][0][0];
    
    #pragma unroll 4
    for(int i=0; i<27; ++i) {
        dst[tid + i * 256] = src[tid + i * 256];
    }
    __syncthreads();

    // 2. Compute Mapping
    // tid (0..255) -> [c_out (16)][h_out (4)][w_out (4)]
    int w_out = tid % 4;
    int h_out = (tid / 4) % 4;
    int c_out = tid / 16;

    float v[4];
    #pragma unroll
    for(int i=0; i<4; ++i) v[i] = 0.0f;

    long long out_base = (long long)b * 2048 + tid; // [Batch, Time, 256]

    // 3. Time Loop
    for (int t = 0; t < 8; ++t) {
        float max_spike = 0.0f;

        // 2x2 Pooling Window
        #pragma unroll
        for (int py = 0; py < 2; ++py) {
            #pragma unroll
            for (int px = 0; px < 2; ++px) {
                int sub_idx = py * 2 + px;
                
                // Init Current with Bias
                float I = c_conv2_b[c_out];
                
                int h_base = h_out * 2 + py;
                int w_base = w_out * 2 + px;

                // Convolution 5x5 x 6 Input Channels
                #pragma unroll
                for (int kh = 0; kh < 5; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < 5; ++kw) {
                        // Loop Input Channels
                        // Weights linear index: c_out*150 + ic*25 + ...
                        int w_base_idx = c_out * 150 + kh * 5 + kw;
                        
                        #pragma unroll
                        for (int ic = 0; ic < 6; ++ic) {
                            float val = s_spikes[t][ic][h_base + kh][w_base + kw];
                            I += val * c_conv2_w[w_base_idx + ic * 25];
                        }
                    }
                }

                // LIF
                float voltage = v[sub_idx] + I;
                float spike = (voltage >= 1.0f) ? 1.0f : 0.0f;
                
                if (spike > 0.5f) v[sub_idx] = 0.0f;
                else v[sub_idx] = voltage;

                if (spike > max_spike) max_spike = spike;
            }
        }
        
        // Write Output: [Batch, Time, FlatIndex]
        // Stride for time is 256
        out_spikes[out_base + t * 256] = max_spike;
    }
}

// FC Layers
__global__ void k_fc1_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N
) {
    int n = blockIdx.x; 
    int oid = threadIdx.x; 
    if (oid >= 120) return;

    float b_val = bias[oid];
    float v = 0.0f;

    const float* in_ptr = input + (long long)n * 2048;
    float* out_ptr = output + (long long)n * 960;      
    const float* w_ptr = weight + oid * 256;

    for(int t=0; t<8; ++t) {
        float sum = b_val;
        const float* in_t = in_ptr + t * 256;
        
        // Unrolling hint
        #pragma unroll 8
        for(int i=0; i<256; ++i) {
            sum += in_t[i] * w_ptr[i];
        }
        
        float voltage = v + sum;
        float spike = (voltage >= 1.0f) ? 1.0f : 0.0f;
        v = (spike > 0.5f) ? 0.0f : voltage;
        
        out_ptr[t * 120 + oid] = spike;
    }
}

__global__ void k_fc2_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N
) {
    int n = blockIdx.x; 
    int oid = threadIdx.x;
    if (oid >= 84) return;

    float b_val = bias[oid];
    float v = 0.0f;

    const float* in_ptr = input + (long long)n * 960;
    float* out_ptr = output + (long long)n * 672;    
    const float* w_ptr = weight + oid * 120;

    for(int t=0; t<8; ++t) {
        float sum = b_val;
        const float* in_t = in_ptr + t * 120;
        
        #pragma unroll 8
        for(int i=0; i<120; ++i) sum += in_t[i] * w_ptr[i];
        
        float voltage = v + sum;
        float spike = (voltage >= 1.0f) ? 1.0f : 0.0f;
        v = (spike > 0.5f) ? 0.0f : voltage;
        
        out_ptr[t * 84 + oid] = spike;
    }
}

__global__ void k_fc3_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N
) {
    int n = blockIdx.x; 
    int oid = threadIdx.x; 
    if (oid >= 10) return;

    float sum_all_t = 0.0f;
    float b_val = bias[oid];
    const float* w_ptr = weight + oid * 84;
    const float* in_ptr = input + (long long)n * 672;

    for(int t=0; t<8; ++t) {
        float sum_t = b_val;
        const float* in_t = in_ptr + t * 84;
        for(int i=0; i<84; ++i) sum_t += in_t[i] * w_ptr[i];
        sum_all_t += sum_t; 
    }
    output[n * 10 + oid] = sum_all_t / 8.0f; 
}

// ===================================================================================
// Inference Driver
// ===================================================================================
std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b
) {
    checkCudaErrors(cudaMemcpyToSymbol(c_conv1_w, d_conv1_w, 150 * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv1_b, d_conv1_b, 6 * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv2_w, d_conv2_w, 2400 * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_conv2_b, d_conv2_b, 16 * sizeof(float), 0, cudaMemcpyDeviceToDevice));

    int N_ALL = images.size();
    std::vector<int> predictions(N_ALL);
    int batch_size = 2048;
    int num_batches = (N_ALL + batch_size - 1) / batch_size;

    float *d_s1, *d_s2, *d_s3, *d_s4, *d_out;
    float *d_in;
    
    checkCudaErrors(cudaMalloc(&d_in, batch_size * 784 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_s1, batch_size * 6912 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_s2, batch_size * 2048 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_s3, batch_size * 960 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_s4, batch_size * 672 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_out, batch_size * 10 * sizeof(float)));

    float* h_in;
    float* h_out;
    checkCudaErrors(cudaMallocHost(&h_in, batch_size * 784 * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&h_out, batch_size * 10 * sizeof(float)));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int b = 0; b < num_batches; ++b) {
        int start = b * batch_size;
        int count = std::min(batch_size, N_ALL - start);

        for(int i=0; i<count; ++i) 
            memcpy(h_in + i*784, images[start+i].data(), 784*sizeof(float));
        
        checkCudaErrors(cudaMemcpyAsync(d_in, h_in, count*784*sizeof(float), cudaMemcpyHostToDevice, stream));

        // Kernel Launch Config
        
        // Conv1: 864 threads per block (1 block per image)
        k_conv1_max_parallel<<<count, dim3(12,12,6), 0, stream>>>(d_in, d_s1, count);
        
        // Conv2: 256 threads per block (1 block per image)
        k_conv2_max_parallel<<<count, 256, 0, stream>>>(d_s1, d_s2, count);
        
        // FC Layers: 1 block per image
        k_fc1_opt<<<count, 128, 0, stream>>>(d_s2, d_fc1_w, d_fc1_b, d_s3, count);
        k_fc2_opt<<<count, 128, 0, stream>>>(d_s3, d_fc2_w, d_fc2_b, d_s4, count);
        k_fc3_opt<<<count, 32, 0, stream>>>(d_s4, d_fc3_w, d_fc3_b, d_out, count);

        checkCudaErrors(cudaMemcpyAsync(h_out, d_out, count*10*sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));

        for(int i=0; i<count; ++i) {
            float* p = h_out + i*10;
            predictions[start+i] = std::distance(p, std::max_element(p, p+10));
        }
    }

    checkCudaErrors(cudaFree(d_in)); checkCudaErrors(cudaFree(d_s1));
    checkCudaErrors(cudaFree(d_s2)); checkCudaErrors(cudaFree(d_s3));
    checkCudaErrors(cudaFree(d_s4)); checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFreeHost(h_in)); checkCudaErrors(cudaFreeHost(h_out));
    checkCudaErrors(cudaStreamDestroy(stream));

    return predictions;
}

// ===================================================================================
// Main
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    std::string dir = argv[1];
    
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty()) {
         images = read_mnist_images(dir + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
         labels = read_mnist_labels(dir + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    }
    if (images.empty()) return 1;

    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
        );
    
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    checkCudaErrors(cudaFree(d_conv1_w)); checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w)); checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));   checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));   checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));   checkCudaErrors(cudaFree(d_fc3_b));
    
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) correct_predictions++;
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    return 0;
}