# Qwen3-1.7B 微调模型推理服务
# 模型: JohnGuo/Qwen3-1.7B-finetuned (FP16, ~3.8GB)
# 适用于 5090 (需要 CUDA 12.8+)

# for GPU
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.0

# 清除原有的ENTRYPOINT,非常重要，避免CMD被覆盖
ENTRYPOINT []

# 设置工作目录
WORKDIR /app

# 先复制项目中的所有文件
COPY . .

# 复制依赖文件并安装依赖
# 这一步单独做可以利用Docker的层缓存机制，如果requirements.txt不变，则不会重新安装
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir -r requirements.txt

# 下载模型
RUN python3 download_model.py

# 声明容器对外暴露的端口
EXPOSE 8000

# 容器启动时运行的命令
# 使用uvicorn启动FastAPI服务，并监听所有网络接口的8000端口
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
