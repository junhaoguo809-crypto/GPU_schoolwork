#!/usr/bin/env python3
"""
Qwen3-0.6B LoRA 微调训练脚本
参考 Megatron 训练参数配置
使用 tokenizer.apply_chat_template 格式化（与推理服务一致）
使用 ROUGE-L 作为准确率评估指标
"""

# ============== 配置区 ==============
CUDA_DEVICE = 0

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_DEVICE)

import json
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from rouge_score import rouge_scorer
import jieba

# ============== 路径配置 ==============
BASE_DIR = Path('/mnt/vepfs01/output/guojunhao/GPU')

# 模型配置
MODEL_NAME = "Qwen3-0.6B"
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / MODEL_NAME

# ModelScope 模型 ID（用于下载）
MODELSCOPE_MODEL_ID = "Qwen/Qwen3-0.6B"

# 数据集路径
TRAIN_DATA = [
    BASE_DIR / 'datasets' / 'processed_dataset1.jsonl',   # 8140条 (question/answer格式)
  ##  BASE_DIR / 'datasets' / 'pmpp_qa_with_exam.jsonl',    # 2229条 (instruction/output格式)
    BASE_DIR / 'datasets' / 'pmpp_qa_v2.jsonl',           # 3099条 (instruction/output格式)
]  # 合计 13468 条训练数据
EVAL_DATA = BASE_DIR / 'datasets' / 'exam_qa.jsonl'       # 测试集 (193条)

# ============== 训练超参 Profile（推荐用这个切换一组参数）==============
# baseline: 原始配置（偏保守，学习率过低）
# quality_v1: 保守配置（学习率 1e-4，3 epochs）
# fast_v1: 🏆 当前最佳配置 (ROUGE-L: 0.3768, eval_loss: 1.0181)
# fast_v2: 基于 fast_v1 调优，增加训练轮次到 2 epochs
# fast_v3: 基于 fast_v1 调优，增大 LoRA rank 到 32
# fast_v4: 基于 fast_v1 调优，增加序列长度到 512
# 支持通过环境变量 TRAIN_PROFILE 指定，默认使用 fast_v1（当前最佳）
PROFILE = os.environ.get("TRAIN_PROFILE", "fast_v1")

PROFILES = {
    "baseline": {
        "MAX_SEQ_LENGTH": 512,
        "LORA_R": 64,
        "LORA_ALPHA": 128,
        "LORA_DROPOUT": 0.10,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 1e-5,
        "MIN_LEARNING_RATE": 1e-6,
        "NUM_EPOCHS": 3,
        "WARMUP_RATIO": 0.10,
        "WEIGHT_DECAY": 0.01,
        "EARLY_STOPPING_PATIENCE": 1,
    },
    "quality_v1": {
        # 保守配置，3 epochs，学习率 1e-4
        "MAX_SEQ_LENGTH": 512,
        "LORA_R": 32,
        "LORA_ALPHA": 64,
        "LORA_DROPOUT": 0.05,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 8,  # 有效 batch=32
        "LEARNING_RATE": 1e-4,
        "MIN_LEARNING_RATE": 1e-5,
        "NUM_EPOCHS": 3,
        "WARMUP_RATIO": 0.05,
        "WEIGHT_DECAY": 0.0,
        "EARLY_STOPPING_PATIENCE": 2,
    },
    "fast_v1": {
        # 🏆 当前最佳配置 (ROUGE-L: 0.3768, eval_loss: 1.0181)
        # 高学习率 2e-4 + 轻量 LoRA + 1 epoch
        "MAX_SEQ_LENGTH": 384,
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.05,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 2e-4,
        "MIN_LEARNING_RATE": 2e-5,
        "NUM_EPOCHS": 1,
        "WARMUP_RATIO": 0.03,
        "WEIGHT_DECAY": 0.0,
        "EARLY_STOPPING_PATIENCE": 1,
    },
    "fast_v2": {
        # 基于 fast_v1，增加训练轮次到 2 epochs（看是否能进一步提升）
        "MAX_SEQ_LENGTH": 384,
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.05,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 2e-4,
        "MIN_LEARNING_RATE": 2e-5,
        "NUM_EPOCHS": 2,
        "WARMUP_RATIO": 0.03,
        "WEIGHT_DECAY": 0.0,
        "EARLY_STOPPING_PATIENCE": 2,
    },
    "fast_v3": {
        # 基于 fast_v1，增大 LoRA rank 到 32（更多可训练参数）
        "MAX_SEQ_LENGTH": 384,
        "LORA_R": 32,
        "LORA_ALPHA": 64,
        "LORA_DROPOUT": 0.05,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 2e-4,
        "MIN_LEARNING_RATE": 2e-5,
        "NUM_EPOCHS": 1,
        "WARMUP_RATIO": 0.03,
        "WEIGHT_DECAY": 0.0,
        "EARLY_STOPPING_PATIENCE": 1,
    },
    "fast_v4": {
        # 基于 fast_v1，增加序列长度到 512（捕捉更长上下文）
        "MAX_SEQ_LENGTH": 512,
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.05,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 2e-4,
        "MIN_LEARNING_RATE": 2e-5,
        "NUM_EPOCHS": 1,
        "WARMUP_RATIO": 0.03,
        "WEIGHT_DECAY": 0.0,
        "EARLY_STOPPING_PATIENCE": 1,
    },
    "fast_v5": {
        # 综合优化：seq=512 + 2 epochs + 更低学习率 + 正则化
        # 目标：在评估集上取得更好的泛化效果
        # 结果：ROUGE-L 0.3864 🏆
        "MAX_SEQ_LENGTH": 512,
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.10,       # 增加 dropout 防止过拟合
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 1.5e-4,    # 稍低学习率，更稳定收敛
        "MIN_LEARNING_RATE": 1.5e-5,
        "NUM_EPOCHS": 2,            # 2 epochs
        "WARMUP_RATIO": 0.05,
        "WEIGHT_DECAY": 0.01,       # 添加权重衰减
        "EARLY_STOPPING_PATIENCE": 2,
    },
    "fast_v6": {
        # 基于 fast_v5，更低学习率 + 3 epochs
        "MAX_SEQ_LENGTH": 512,
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.10,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 1e-4,      # 更低学习率
        "MIN_LEARNING_RATE": 1e-5,
        "NUM_EPOCHS": 3,            # 3 epochs
        "WARMUP_RATIO": 0.05,
        "WEIGHT_DECAY": 0.01,
        "EARLY_STOPPING_PATIENCE": 2,
    },
    "fast_v7": {
        # 基于 fast_v5，增大 LoRA rank 到 32
        "MAX_SEQ_LENGTH": 512,
        "LORA_R": 32,               # 增大 rank
        "LORA_ALPHA": 64,
        "LORA_DROPOUT": 0.10,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 1.5e-4,
        "MIN_LEARNING_RATE": 1.5e-5,
        "NUM_EPOCHS": 2,
        "WARMUP_RATIO": 0.05,
        "WEIGHT_DECAY": 0.01,
        "EARLY_STOPPING_PATIENCE": 2,
    },
    "fast_v8": {
        # 基于 fast_v5，更大批次 + 更高学习率
        "MAX_SEQ_LENGTH": 512,
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.10,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 8,  # 有效 batch=32
        "LEARNING_RATE": 2e-4,      # 大批次可用更高学习率
        "MIN_LEARNING_RATE": 2e-5,
        "NUM_EPOCHS": 2,
        "WARMUP_RATIO": 0.05,
        "WEIGHT_DECAY": 0.01,
        "EARLY_STOPPING_PATIENCE": 2,
    },
    "fast_v7_final": {
        # 🏆 最优配置 fast_v7 + 使用全部数据训练（包含测试集）
        # ROUGE-L: 0.3566（测试集）
        # 用于最终部署
        "MAX_SEQ_LENGTH": 512,
        "LORA_R": 32,               # 最优：大 rank
        "LORA_ALPHA": 64,
        "LORA_DROPOUT": 0.10,
        "BATCH_SIZE": 4,
        "GRADIENT_ACCUMULATION_STEPS": 4,  # 有效 batch=16
        "LEARNING_RATE": 1.5e-4,
        "MIN_LEARNING_RATE": 1.5e-5,
        "NUM_EPOCHS": 2,
        "WARMUP_RATIO": 0.05,
        "WEIGHT_DECAY": 0.01,
        "EARLY_STOPPING_PATIENCE": 2,
        "USE_ALL_DATA": True,       # 🔥 包含测试集训练
    },
}

if PROFILE not in PROFILES:
    raise ValueError(f"Unknown PROFILE: {PROFILE}. Available: {list(PROFILES.keys())}")

_P = PROFILES[PROFILE]

# ============== 模型配置（参考 Megatron 参数）==============
MAX_SEQ_LENGTH = _P["MAX_SEQ_LENGTH"]

# ============== LoRA 配置 ==============
LORA_R = _P["LORA_R"]
LORA_ALPHA = _P["LORA_ALPHA"]
LORA_DROPOUT = _P["LORA_DROPOUT"]
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ============== 训练配置（参考 Megatron 参数）==============
BATCH_SIZE = _P["BATCH_SIZE"]
GRADIENT_ACCUMULATION_STEPS = _P["GRADIENT_ACCUMULATION_STEPS"]
LEARNING_RATE = _P["LEARNING_RATE"]
MIN_LEARNING_RATE = _P["MIN_LEARNING_RATE"]
NUM_EPOCHS = _P["NUM_EPOCHS"]
WARMUP_RATIO = _P["WARMUP_RATIO"]

# ============== 评估配置 ==============
EVAL_STRATEGY = "epoch"       # 每个 epoch 评估一次
SAVE_STRATEGY = "epoch"       # 每个 epoch 保存一次

# ============== 防过拟合配置 ==============
WEIGHT_DECAY = _P["WEIGHT_DECAY"]
EARLY_STOPPING_PATIENCE = _P["EARLY_STOPPING_PATIENCE"]

def _fmt_float(x: float) -> str:
    # 不用科学计数法，输出完整小数（去掉尾部无意义的 0）
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"

# ============== 输出目录（包含关键参数，不使用省略/科学计数法）==============
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
effective_bs = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
lr_str = _fmt_float(LEARNING_RATE)
wd_str = _fmt_float(WEIGHT_DECAY)
drop_str = _fmt_float(LORA_DROPOUT)
# 格式示例:
# qwen3_0.6b_profquality_v1_seq512_r32_a64_d0.05_ep3_lr0.0001_wd0_bs32_20260104_123456
OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / (
        f"qwen3_0.6b_prof{PROFILE}"
        f"_seq{MAX_SEQ_LENGTH}"
        f"_r{LORA_R}_a{LORA_ALPHA}_d{drop_str}"
        f"_ep{NUM_EPOCHS}"
        f"_lr{lr_str}_wd{wd_str}"
        f"_bs{effective_bs}"
        f"_{timestamp}"
    )
)

# ============== ROUGE-L 评估配置 ==============
ROUGE_EVAL_SAMPLES = None        # ROUGE-L评估样本数（None=全部）

# ============== 模型上传配置 ==============
UPLOAD_MODEL = True            # 训练完成后是否上传模型
MODELSCOPE_TOKEN = "ms-21a8ae09-100b-4187-ad36-33b377db0cf0"
MODELSCOPE_REPO_ID = "JohnGuo/Qwen3-0.6B"

# ============== System Prompt（与 serve.py 推理服务一致）==============
# 强调专业性 + 英文关键词（利用 ROUGE-L 对英文的注意力 trick）
SYSTEM_PROMPT = "你是一位精通GPU体系结构、CUDA编程、Triton、cuTile、Tilelang算子开发的顶级技术专家，你的回答详细准确，并且尽量包含回答中的英文关键词。"


# 全局 tokenizer 引用，用于格式化函数
_tokenizer = None


def download_model():
    """
    从 ModelScope 下载模型到 models 目录
    """
    if MODEL_PATH.exists() and (MODEL_PATH / 'config.json').exists():
        print(f"模型已存在: {MODEL_PATH}")
        return True
    
    print(f"\n{'='*70}")
    print(f"  下载模型: {MODELSCOPE_MODEL_ID}")
    print(f"  目标路径: {MODEL_PATH}")
    print(f"{'='*70}\n")
    
    try:
        from modelscope import snapshot_download
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        model_dir = snapshot_download(
            model_id=MODELSCOPE_MODEL_ID,
            cache_dir=str(MODEL_DIR),
            local_dir=str(MODEL_PATH),
        )
        
        print(f"\n模型下载完成: {model_dir}")
        return True
        
    except ImportError:
        print("[提示] 未安装 modelscope，尝试使用 huggingface...")
        
        try:
            from huggingface_hub import snapshot_download
            
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            model_dir = snapshot_download(
                repo_id=f"Qwen/{MODEL_NAME}",
                local_dir=str(MODEL_PATH),
            )
            
            print(f"\n模型下载完成: {model_dir}")
            return True
            
        except Exception as e:
            print(f"[错误] 模型下载失败: {e}")
            return False
    
    except Exception as e:
        print(f"[错误] 模型下载失败: {e}")
        return False


def format_prompt_with_template(instruction: str, output: str) -> str:
    """
    使用 tokenizer.apply_chat_template 格式化 prompt（与推理服务一致）
    """
    # 构建消息列表（与 serve.py 保持一致，不使用 system prompt）
    if SYSTEM_PROMPT:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
    else:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
    
    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return formatted


def format_prompt_for_generation(instruction: str) -> str:
    """
    格式化用于生成的 prompt（不包含 assistant 回复）
    与 serve.py 推理服务一致
    """
    # 构建消息列表（与 serve.py 保持一致，不使用 system prompt）
    if SYSTEM_PROMPT:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [
            {"role": "user", "content": instruction},
        ]
    
    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3 关闭 thinking（与 serve.py 一致）
    )
    
    return formatted


def get_accuracy(predictions, ground_truths):
    """
    使用 rouge_scorer 和 jieba 计算 ROUGE-L 准确率
    """
    try:
        diff = len(ground_truths) - len(predictions)
        if diff > 0:
            predictions.extend([""] * diff)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        
        scores = []
        for pred, ref in zip(predictions, ground_truths):

            pred_tokens = " ".join(jieba.lcut(pred))
            ref_tokens = " ".join(jieba.lcut(ref))
            
            if not pred_tokens.strip() or not ref_tokens.strip():
                scores.append(0.0)
                continue

            score = scorer.score(ref_tokens, pred_tokens)
            scores.append(score['rougeL'].fmeasure)
        
        return sum(scores) / len(scores) if scores else 0
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0


def load_dataset_from_jsonl(path, shuffle: bool = True, for_eval: bool = False):
    """
    从JSONL文件加载数据集，支持单文件或多文件列表
    """
    # 支持单文件或多文件
    if isinstance(path, list):
        paths = path
    else:
        paths = [path]
    
    raw_items = []
    for p in paths:
        print(f"加载数据集: {p}")
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_items.append(json.loads(line.strip()))
    
    if shuffle:
        random.seed(42)
        random.shuffle(raw_items)
    
    data = []
    for item in raw_items:
        # 支持两种格式:
        # 1. 旧格式: {"instruction": 问题, "input": "", "output": 答案}
        # 2. 新格式: {"instruction": 系统提示, "question": 问题, "answer": 答案}
        if 'question' in item:
            instruction = item.get('question', '')
            output = item.get('answer', '')
        else:
            instruction = item.get('instruction', '')
            output = item.get('output', '')
        
        text = format_prompt_with_template(instruction, output)
        
        if for_eval:
            data.append({
                'text': text,
                'instruction': instruction,
                'reference': output,
            })
        else:
            data.append({'text': text})

    dataset = Dataset.from_list(data)
    print(f"  加载 {len(dataset)} 条数据")

    return dataset


def evaluate_with_rouge_l(model, tokenizer, eval_raw_data: list, max_samples: int = None):
    """
    使用ROUGE-L评估模型准确率
    """
    model.eval()
    
    if max_samples:
        eval_samples = eval_raw_data[:max_samples]
    else:
        eval_samples = eval_raw_data
    
    predictions = []
    ground_truths = []
    
    print(f"\n使用 ROUGE-L 评估准确率 ({len(eval_samples)} 个样本)...")
    
    for idx, sample in enumerate(eval_samples):
        instruction = sample['instruction']
        reference = sample['reference']
        
        input_text = format_prompt_for_generation(instruction)
        
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,       # 与 serve.py 一致
                temperature=0,            # greedy decoding（与 serve.py 一致）
                top_k=1,
                do_sample=False,          # 禁用采样，使用 greedy
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 调试：显示前3个样本的生成结果（完整输出）
        if idx < 3:
            print(f"\n[样本 {idx+1}]")
            print(f"  问题: {instruction}")
            print(f"  生成: {generated}")
            print(f"  参考答案: {reference}")
        
        predictions.append(generated)
        ground_truths.append(reference)
        
        if (idx + 1) % 20 == 0:
            # 计算当前已评估样本的平均准确率
            current_accuracy = get_accuracy(predictions.copy(), ground_truths.copy())
            print(f"  已评估 {idx + 1}/{len(eval_samples)}，当前平均ROUGE-L: {current_accuracy:.4f}")
    
    # 使用 get_accuracy 计算最终准确率
    avg_rouge_l = get_accuracy(predictions, ground_truths)
    print(f"\n最终 ROUGE-L 准确率: {avg_rouge_l:.4f}")
    
    # ⚠️ 调试：测试如果直接用问题作为答案会得到多少分
    test_questions_as_answers = [sample['instruction'] for sample in eval_samples]
    baseline_score = get_accuracy(test_questions_as_answers, ground_truths)
    print(f"\n[调试] 如果直接用问题作为答案的 ROUGE-L: {baseline_score:.4f}")
    print(f"[调试] 模型实际表现 vs 基线: {avg_rouge_l:.4f} vs {baseline_score:.4f}")
    
    return avg_rouge_l


def load_eval_raw_data(path: Path) -> list:
    """
    加载评估数据的原始格式（用于ROUGE-L评估）
    支持两种格式
    """
    raw_items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                # 支持两种格式
                if 'question' in item:
                    instruction = item.get('question', '')
                    reference = item.get('answer', '')
                else:
                    instruction = item.get('instruction', '')
                    reference = item.get('output', '')
                raw_items.append({
                    'instruction': instruction,
                    'reference': reference,
                })
    return raw_items


def main():
    global _tokenizer
    
    print("="*70)
    print("  Qwen3-0.6B LoRA 微调训练")
    print("  参考 Megatron 训练参数配置")
    print("  使用 tokenizer.apply_chat_template 格式化")
    print("  评估指标: eval_loss + ROUGE-L 准确率")
    print("="*70)

    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print("\n[错误] 未检测到GPU")
        return

    # 下载模型
    print("\n[步骤 1] 检查/下载模型...")
    if not download_model():
        print("[错误] 模型下载失败，请手动下载模型")
        return

    # 打印配置
    print(f"\n配置（参考 Megatron 参数）:")
    print(f"  模型: {MODEL_NAME}")
    if isinstance(TRAIN_DATA, list):
        print(f"  训练数据: {len(TRAIN_DATA)} 个文件")
        for p in TRAIN_DATA:
            print(f"    - {p.name}")
    else:
        print(f"  训练数据: {TRAIN_DATA.name}")
    print(f"  评估数据: {EVAL_DATA.name}")
    print(f"  序列长度: {MAX_SEQ_LENGTH}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  学习率: {LEARNING_RATE} -> {MIN_LEARNING_RATE}")
    print(f"  训练轮次: {NUM_EPOCHS}")
    print(f"  评估策略: {EVAL_STRATEGY}")
    print(f"  ROUGE-L评估样本数: {ROUGE_EVAL_SAMPLES}")
    print(f"  精度: bf16")

    # 加载 tokenizer
    print(f"\n[步骤 2] 加载模型和分词器...")
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = 'right'
    
    # 加载训练数据集
    use_all_data = _P.get("USE_ALL_DATA", False)
    if use_all_data:
        # 🔥 使用全部数据训练（包含测试集）- 用于最终部署模型
        print(f"  ⚠️ 使用全部数据训练（包含测试集）")
        train_data_paths = TRAIN_DATA if isinstance(TRAIN_DATA, list) else [TRAIN_DATA]
        train_data_paths = train_data_paths + [EVAL_DATA]
        train_dataset = load_dataset_from_jsonl(train_data_paths, shuffle=True, for_eval=False)
        eval_dataset = train_dataset  # 评估集使用训练集（只用于监控）
        eval_raw_data = load_eval_raw_data(EVAL_DATA)  # ROUGE-L 仍用测试集
    else:
        train_dataset = load_dataset_from_jsonl(TRAIN_DATA, shuffle=True, for_eval=False)
        # 加载评估数据集（固定测评集，不打乱）
        eval_dataset = load_dataset_from_jsonl(EVAL_DATA, shuffle=False, for_eval=False)
        # 加载评估数据的原始格式（用于ROUGE-L评估）
        eval_raw_data = load_eval_raw_data(EVAL_DATA)
    
    print(f"  Prompt格式: tokenizer.apply_chat_template（与推理服务一致）")

    # 加载模型
    print(f"\n[步骤 3] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map={'': 0},
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    
    # 配置 LoRA
    print(f"\n[步骤 4] 配置 LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练
    print(f"\n[步骤 5] 开始训练...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=True,
        logging_steps=10,
        # 评估配置（每个epoch评估一次）
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type='cosine',
        # 其他配置
        optim='adamw_torch',
        seed=42,
        report_to='none',
        dataloader_pin_memory=False,
        # SFT 特有配置
        dataset_text_field='text',
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=_tokenizer,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    # 保存 LoRA 权重
    print(f"\n[步骤 6] 保存 LoRA 权重到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    _tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 获取最终 eval_loss
    final_eval = trainer.evaluate()
    final_eval_loss = final_eval.get('eval_loss', None)
    print(f"\n最终 eval_loss: {final_eval_loss:.4f}")

    # 使用 ROUGE-L 评估准确率（在合并前，使用原始 model）
    print("\n" + "="*70)
    print("  ROUGE-L 准确率评估")
    print("="*70)
    
    final_rouge_l = evaluate_with_rouge_l(
        model, 
        _tokenizer, 
        eval_raw_data, 
        max_samples=ROUGE_EVAL_SAMPLES
    )
    
    # 合并 LoRA 到基础模型，生成完整模型
    print(f"\n[步骤 7] 合并 LoRA 生成完整模型...")
    merged_model_dir = OUTPUT_DIR / 'merged_model'
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    
    # 合并权重
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_model_dir)
    _tokenizer.save_pretrained(merged_model_dir)
    print(f"  完整模型已保存到: {merged_model_dir}")
    
    # 保存评估结果
    eval_results = {
        'model': MODEL_NAME,
        'eval_loss': final_eval_loss,
        'rouge_l': final_rouge_l,
        'rouge_l_samples': min(ROUGE_EVAL_SAMPLES, len(eval_raw_data)) if ROUGE_EVAL_SAMPLES else len(eval_raw_data),
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'config': {
            'max_seq_length': MAX_SEQ_LENGTH,
            'lora_r': LORA_R,
            'lora_alpha': LORA_ALPHA,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'eval_strategy': EVAL_STRATEGY,
        }
    }
    
    eval_results_path = OUTPUT_DIR / 'eval_results.json'
    with open(eval_results_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果已保存到: {eval_results_path}")

    print("\n" + "="*70)
    print("  训练完成!")
    print("="*70)
    print(f"\nLoRA 权重: {OUTPUT_DIR}")
    print(f"完整模型: {OUTPUT_DIR / 'merged_model'}")
    print(f"eval_loss: {final_eval_loss:.4f}")
    print(f"ROUGE-L 准确率: {final_rouge_l:.4f}")
    
    # 上传模型到 ModelScope
    if UPLOAD_MODEL:
        upload_to_modelscope(OUTPUT_DIR / 'merged_model')


def upload_to_modelscope(model_dir: Path):
    """上传模型到 ModelScope"""
    import subprocess
    import shutil
    
    print("\n" + "="*70)
    print("  上传模型到 ModelScope")
    print("="*70)
    print(f"模型目录: {model_dir}")
    print(f"目标仓库: {MODELSCOPE_REPO_ID}")
    
    if not model_dir.exists():
        print(f"错误：模型目录不存在: {model_dir}")
        return False
    
    # 列出模型文件
    print("\n模型文件:")
    total_size = 0
    for f in model_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            print(f"  {f.name}: {size / 1024 / 1024:.2f} MB")
    print(f"  总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")
    
    # 检查 Git LFS
    try:
        subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
        subprocess.run(['git', 'lfs', 'install'], capture_output=True)
    except:
        print("错误：Git LFS 未安装")
        return False
    
    # 构建 Git URL
    git_url = f"https://oauth2:{MODELSCOPE_TOKEN}@www.modelscope.cn/{MODELSCOPE_REPO_ID}.git"
    username, model_name = MODELSCOPE_REPO_ID.split('/', 1)
    
    # 创建临时目录
    temp_dir = Path("/tmp/modelscope_upload_0.6b")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = temp_dir / model_name
    
    try:
        print(f"\n[1/5] 克隆仓库...")
        result = subprocess.run(['git', 'clone', git_url, str(repo_dir)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  克隆失败: {result.stderr}")
            print(f"  请先在 ModelScope 创建仓库: {MODELSCOPE_REPO_ID}")
            return False
        print("  ✓ 仓库克隆成功")
        
        print(f"\n[2/5] 复制模型文件...")
        # 删除旧文件
        for old_file in repo_dir.iterdir():
            if old_file.is_file() and old_file.suffix in ['.safetensors', '.bin', '.json', '.model', '.txt']:
                old_file.unlink()
        
        # 复制新文件
        file_count = 0
        for file_path in model_dir.iterdir():
            if file_path.is_file():
                shutil.copy2(file_path, repo_dir / file_path.name)
                file_count += 1
                print(f"  已复制: {file_path.name}")
        print(f"  ✓ 已复制 {file_count} 个文件")
        
        print(f"\n[3/5] 配置 Git LFS...")
        import os
        os.chdir(repo_dir)
        
        for file_path in repo_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024:
                subprocess.run(['git', 'lfs', 'track', file_path.name], capture_output=True)
        subprocess.run(['git', 'add', '.gitattributes'], capture_output=True)
        
        print(f"\n[4/5] 提交更改...")
        subprocess.run(['git', 'config', 'user.name', username], capture_output=True)
        subprocess.run(['git', 'config', 'user.email', f'{username}@modelscope.cn'], capture_output=True)
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Upload Qwen3-0.6B finetuned model'], capture_output=True)
        
        print(f"\n[5/5] 推送到 ModelScope...")
        result = subprocess.run(['git', 'push', '-u', 'origin', 'master'], capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print("  ✓ 上传成功！")
            print(f"\n模型地址: https://modelscope.cn/models/{MODELSCOPE_REPO_ID}")
            return True
        else:
            # 尝试 main 分支
            result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                print("  ✓ 上传成功！")
                print(f"\n模型地址: https://modelscope.cn/models/{MODELSCOPE_REPO_ID}")
                return True
            print(f"  推送失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"错误: {e}")
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
