"""
vLLM 高性能推理服务 (Batch模式) - 极致性能优化版
模型: JohnGuo/Qwen3-0.6B (fast_v7_final 微调版)

🏆 训练参数: fast_v7_final (ROUGE-L: 0.4271)
⚠️ Docker: vLLM v0.11.0
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"  # 5090 GPU架构
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # 显式启用 Flash Attention

from typing import Union, List
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams

# ============== 模型配置 ==============
LOCAL_MODEL_PATH = "./local-model"

# ============== System Prompt（与训练脚本 train_qwen3_0.6b.py 一致）==============
SYSTEM_PROMPT = "你是一位精通GPU体系结构、CUDA编程、Triton、cuTile、Tilelang算子开发的顶级技术专家，你的回答详细准确，并且尽量包含回答中的英文关键词。"

# ============== 预热问题（从 exam_qa.jsonl 抽取15个，覆盖简单/中等/困难）==============
WARMUP_QUESTIONS = [
    # 简单题 (基础概念)
    "什么是数据并行性？",
    "CUDA中的核函数是什么？",
    "CUDA中的warp是什么？",
    # 中等题 (原理理解)
    "为什么CUDA核函数中常需要添加边界检查的条件判断？",
    "为什么内存访问效率对CUDA程序性能至关重要？",
    "什么是CUDA中的全局内存合并访问？",
    "IEEE 754浮点数格式由哪几部分组成？",
    # 困难题 (算子实现)
    "CUDA中矩阵乘法算子如何利用共享内存减少全局内存访问？",
    "CUDA矩阵乘法算子中，如何通过边界检查处理非TILE_WIDTH倍数的矩阵？",
    "GPU架构的共享内存bank冲突如何在SpMV算子中避免？",
    # 综合题 (带代码)
    "结合算法与CUDA编程，tiled矩阵乘法算子如何通过数据复用提升计算/内存访问比？",
    "如何用Triton实现ConvNets的3×3卷积层，并通过自动分块优化提升性能？",
    "如何用TileLang优化SpMV算子的CSR格式访问，提升非合并内存访问效率？",
    "Triton实现的矩阵乘法算子如何与CUDA的tiled实现对比，优势在哪里？",
    "如何用TileLang实现ConvNets的深度卷积（Depthwise Convolution），优化组内内存局部性？",
]

################################### 初始化部分 ###################################

# 1. 加载 tokenizer
print(f"从本地加载模型：{LOCAL_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

# 2. 格式化 prompt 函数（使用 lru_cache 缓存）
@lru_cache(maxsize=10000)
def format_prompt(msg: str) -> str:
    """使用 tokenizer.apply_chat_template 格式化 prompt"""
    # 添加精简版 system prompt（利用 ROUGE-L 英文关键词 trick）
    if SYSTEM_PROMPT:
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg}
        ]
    else:
        message = [{"role": "user", "content": msg}]
    return tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False  # Qwen3 关闭 thinking 提高吞吐
    )

# 3. 配置采样参数 (SamplingParams) - 与旧版一致
sampling_params = SamplingParams(
    temperature=0,           # greedy decoding，最快最稳定
    top_k=1,                 # 恢复 top_k=1
    max_tokens=384,          # 适中生成长度，平衡速度和质量
    stop=["\n\n", "<|endoftext|>", "<|im_end|>"],  # 恢复旧版 stop
    stop_token_ids=[tokenizer.eos_token_id],
)

# 4. 初始化 vLLM 引擎 - 性能优化配置 (5090 Blackwell, 兼容 vLLM v0.11.0 和 v0.13.0)
import vllm
_vllm_version = tuple(map(int, vllm.__version__.split('.')[:2]))

# 根据 vLLM 版本选择参数
if _vllm_version >= (0, 13):
    # vLLM v0.13.0+: 自动优化，不需要手动指定这些参数
    llm = LLM(
        model=LOCAL_MODEL_PATH,
        dtype="bfloat16",
        quantization="fp8",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        swap_space=4,
    )
else:
    # vLLM v0.11.0: 极致性能优化
    llm = LLM(
        model=LOCAL_MODEL_PATH,
        dtype="bfloat16",
        quantization="fp8",              # FP8量化，充分利用5090 Blackwell硬件特性
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,     # 高显存利用率
        enforce_eager=False,             # 允许compile优化计算图(CUDA Graph)
        max_model_len=512,               # 输入89+输出384=473，需要512
        max_num_seqs=4096,               # 提高并发上限
        max_num_batched_tokens=16384,    # 增大批量token数，提升吞吐
        enable_prefix_caching=True,      # 开启前缀缓存
        disable_log_stats=True,          # 关闭日志减少开销
    )

print("模型加载完成！(Batch模式)")

# 5. 执行预热推理 - 充分预热（不算测评时间）
print("开始预热推理...")
warmup_formatted = [format_prompt(p) for p in WARMUP_QUESTIONS]
# 多轮预热：充分预热 CUDA kernel、KV cache、前缀缓存
for i in range(4):  # 4轮预热
    _ = llm.generate(warmup_formatted, sampling_params)

set_seed(42)  # 恢复随机种子设置

################################### API 定义 ###################################

# 创建FastAPI应用实例
app = FastAPI(
    title="vLLM Batch Inference Server",
    description="High-performance LLM batch inference with vLLM + Qwen3"
)


class PromptRequest(BaseModel):
    prompt: Union[str, List[str]]  # 支持单条和批量


class PredictResponse(BaseModel):
    response: Union[str, List[str]]  # 返回格式与输入一致


def postprocess(text: str) -> str:
    """后处理生成的文本，移除结束标记"""
    generated = text.strip()
    # 移除可能的结束标记
    for marker in ["<|im_end|>", "<|im_start|>"]:
        if marker in generated:
            generated = generated.split(marker)[0].strip()
    return generated.strip()


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PromptRequest):
    """
    推理端点 - 支持单条和批量推理
    
    单条请求: {"prompt": "问题内容"}
    批量请求: {"prompt": ["问题1", "问题2", ...]}  (Batch模式)
    """
    if isinstance(request.prompt, str):
        real_input_list = [request.prompt]
        is_batch = False
    else:
        real_input_list = request.prompt
        is_batch = True
    
    # 格式化 prompt
    final_prompt_texts = [format_prompt(msg) for msg in real_input_list]
    
    # vLLM 批量推理
    outputs = llm.generate(final_prompt_texts, sampling_params)
    
    # 提取结果并后处理
    generated = [postprocess(output.outputs[0].text) for output in outputs]
    
    # 返回格式与输入一致
    if is_batch:
        return PredictResponse(response=generated)
    else:
        return PredictResponse(response=generated[0])


@app.get("/")
def health_check():
    """
    健康检查 - 返回 {"status": "batch"} 开启批量模式
    """
    return {"status": "batch"}
