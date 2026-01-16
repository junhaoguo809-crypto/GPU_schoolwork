#!/bin/bash
# 构建 Docker 镜像
# 使用 --network=host 确保构建时能访问网络下载模型

set -e

IMAGE_NAME="my-vllm11"

echo "=========================================="
echo "构建 Docker 镜像: ${IMAGE_NAME}"
echo "=========================================="

docker build --network=host -t ${IMAGE_NAME} .

echo ""
echo "构建完成！"
echo ""
echo "运行命令："
echo "  docker run -d --gpus all -p 8000:8000 --name vllm11 ${IMAGE_NAME}"
