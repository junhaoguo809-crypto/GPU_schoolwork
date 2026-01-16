"""
download_model.py
使用 ModelScope 下载微调后的 Qwen3-0.6B 模型 (FP16, ~1.4GB)
模型仓库: JohnGuo/Qwen3-0.6B
"""
import os
from modelscope import snapshot_download

# ============== 配置 ==============
# ModelScope 模型 ID (微调后的模型)
MODEL_ID = "JohnGuo/Qwen3-0.6B"

# 本地模型路径
LOCAL_MODEL_PATH = "./local-model"


def download_model():
    """从 ModelScope 下载模型"""
    print("=" * 50)
    print(f"开始下载模型 {MODEL_ID} 到 {LOCAL_MODEL_PATH}...")
    print("=" * 50)

    try:
        # 使用 local_dir 参数直接下载到指定目录
        model_dir = snapshot_download(
            model_id=MODEL_ID,
            local_dir=LOCAL_MODEL_PATH,
            revision="master"
        )
        print(f"ModelScope 返回路径: {model_dir}")

        # 列出下载的文件
        abs_path = os.path.abspath(LOCAL_MODEL_PATH)
        print(f"\n下载完成! 路径: {abs_path}")
        print("文件列表:")
        for f in sorted(os.listdir(abs_path)):
            fpath = os.path.join(abs_path, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  {f}: {size_mb:.2f} MB")

        return abs_path

    except Exception as e:
        print(f"下载错误: {e}")
        raise RuntimeError(f"模型下载失败: {e}")


if __name__ == "__main__":
    model_path = download_model()
    print(f"\n模型准备就绪! 路径: {model_path}")
