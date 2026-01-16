#!/usr/bin/env python3
"""
PMPP教材问答对生成器 v2
改进版：Few-shot Learning + 质量过滤 + 智能分块
"""

import re
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent / "@CLAUDECODE/tasks/1.improve_qa_dataset_generation/docs"))

from few_shot_examples import get_examples_by_difficulty, format_examples_for_prompt
from prompt_templates import build_system_prompt, build_user_prompt, DIFFICULTY_CONFIG
from content_processor import smart_chunk, extract_chapter_content, clean_content
from quality_filter import filter_qa_batch, normalize_qa_format

# ============== 配置 ==============
QWEN_API_KEY = "sk-fe967e0c4ea04a9bbb1a125280d70d6e"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置
MODELS = ["qwen-max", "qwen-plus", "qwen-max-latest"]
current_model_idx = 0

# 章节配置
CHAPTERS = {
    1: {"title": "Introduction", "line_range": (527, 759), "target": 60},
    2: {"title": "Data Parallel Computing", "line_range": (759, 1227), "target": 80},
    3: {"title": "Scalable Parallel Execution", "line_range": (1227, 1703), "target": 80},
    4: {"title": "Memory and Data Locality", "line_range": (1703, 2252), "target": 100},
    5: {"title": "Performance Considerations", "line_range": (2252, 2737), "target": 100},
    6: {"title": "Numerical Considerations", "line_range": (2737, 3064), "target": 60},
    7: {"title": "Parallel Patterns: Convolution", "line_range": (3064, 3625), "target": 80},
    13: {"title": "CUDA Dynamic Parallelism", "line_range": (5548, 6419), "target": 60},
    16: {"title": "Machine Learning", "line_range": (7064, 7629), "target": 70},
    17: {"title": "Computational Thinking", "line_range": (7629, 7891), "target": 50},
    20: {"title": "More on CUDA and GPU", "line_range": (9219, 9394), "target": 60},
}

# 难度分布 (调整后)
DIFFICULTY_DIST = {"easy": 0.50, "medium": 0.30, "hard": 0.20}

# 输出配置
OUTPUT_DIR = Path("/mnt/vepfs01/output/guojunhao/GPU/datasets")
TEXTBOOK_PATH = "/mnt/vepfs01/output/guojunhao/GPU/resourse/MinerU_markdown_PMPP-3rd-Edition_20251224161351_2003740888941424640.md"


def get_model() -> str:
    """获取当前模型"""
    return MODELS[current_model_idx % len(MODELS)]


def switch_model() -> str:
    """切换模型"""
    global current_model_idx
    current_model_idx += 1
    model = MODELS[current_model_idx % len(MODELS)]
    print(f"[切换模型] -> {model}")
    return model


def init_client() -> OpenAI:
    """初始化API客户端"""
    return OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


def load_textbook() -> List[str]:
    """加载教材"""
    with open(TEXTBOOK_PATH, 'r', encoding='utf-8') as f:
        return f.readlines()


def generate_qa_batch(
    client: OpenAI,
    content: str,
    chapter_num: int,
    chapter_title: str,
    difficulty: str,
    batch_size: int = 5,
    retry_count: int = 0
) -> List[Dict]:
    """生成一批问答对"""
    model = get_model()

    # 构建few-shot示例
    examples = get_examples_by_difficulty(difficulty)
    examples_str = format_examples_for_prompt(examples, max_count=3)

    # 构建prompt
    system_prompt = build_system_prompt(difficulty, examples_str, batch_size)
    user_prompt = build_user_prompt(chapter_num, chapter_title, content, difficulty, batch_size)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,  # 降低随机性
            max_tokens=4000
        )

        result = response.choices[0].message.content

        # 解析JSON
        json_match = re.search(r'\[[\s\S]*\]', result)
        if json_match:
            json_str = json_match.group()
            # 清理控制字符
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
            # 修复常见JSON错误
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)

            try:
                qa_pairs = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"  [JSON解析错误] {e}")
                return []

            # 添加元数据
            for qa in qa_pairs:
                qa["chapter"] = chapter_num
                qa["chapter_title"] = chapter_title
                qa["difficulty"] = difficulty

            # 规范化格式
            qa_pairs = [normalize_qa_format(qa) for qa in qa_pairs]

            # 质量过滤
            qa_pairs = filter_qa_batch(qa_pairs, verbose=True)

            return qa_pairs
        else:
            print(f"  [警告] 无法解析JSON响应")
            return []

    except Exception as e:
        error_msg = str(e).lower()
        print(f"  [错误] {e}")

        if any(k in error_msg for k in ["quota", "limit", "rate"]):
            if retry_count < len(MODELS):
                switch_model()
                time.sleep(5)
                return generate_qa_batch(client, content, chapter_num, chapter_title,
                                        difficulty, batch_size, retry_count + 1)

        return []


def generate_chapter_qa(
    client: OpenAI,
    lines: List[str],
    chapter_num: int,
    progress: Dict
) -> List[Dict]:
    """生成单个章节的问答对"""
    config = CHAPTERS[chapter_num]
    target = config["target"]
    chapter_key = str(chapter_num)

    # 检查已有进度
    existing = progress.get("chapter_progress", {}).get(chapter_key, {}).get("generated", 0)
    if existing >= target:
        print(f"  第{chapter_num}章已完成 ({existing}/{target})")
        return []

    print(f"\n{'='*60}")
    print(f"第{chapter_num}章: {config['title']}")
    print(f"目标: {target}条, 已有: {existing}条")
    print(f"{'='*60}")

    # 提取并分块内容
    start, end = config["line_range"]
    raw_content = ''.join(lines[start:end])
    chunks = smart_chunk(raw_content, max_chars=3500)
    print(f"  内容分割为 {len(chunks)} 个语义块")

    chapter_qa = []
    remaining = target - existing

    # 按难度分配
    for difficulty, ratio in DIFFICULTY_DIST.items():
        diff_target = int(remaining * ratio)
        if diff_target == 0:
            continue

        diff_name = DIFFICULTY_CONFIG[difficulty]["name"]
        print(f"\n  生成{diff_name} (目标: {diff_target}条)...")

        generated = 0
        chunk_idx = 0
        max_attempts = len(chunks) * 2

        while generated < diff_target and chunk_idx < max_attempts:
            chunk = chunks[chunk_idx % len(chunks)]
            batch_size = min(5, diff_target - generated)

            print(f"    批次 {generated//5 + 1}: 生成 {batch_size} 条...")

            qa_pairs = generate_qa_batch(
                client, chunk, chapter_num, config["title"],
                difficulty, batch_size
            )

            if qa_pairs:
                chapter_qa.extend(qa_pairs)
                generated += len(qa_pairs)
                print(f"    成功: {len(qa_pairs)}条 (累计: {generated}/{diff_target})")
            else:
                print(f"    本批次无有效输出")

            chunk_idx += 1
            time.sleep(2)

        print(f"  {diff_name}完成: {generated}/{diff_target}")

    # 更新进度
    if chapter_key not in progress.get("chapter_progress", {}):
        progress.setdefault("chapter_progress", {})[chapter_key] = {"generated": 0, "qa_pairs": []}

    progress["chapter_progress"][chapter_key]["generated"] += len(chapter_qa)
    progress["chapter_progress"][chapter_key]["qa_pairs"].extend(chapter_qa)
    progress["all_qa_pairs"].extend(chapter_qa)
    progress["total_generated"] += len(chapter_qa)

    print(f"\n  第{chapter_num}章完成: {len(chapter_qa)}条")
    return chapter_qa


def save_dataset(progress: Dict, output_dir: Path):
    """保存数据集"""
    all_qa = progress.get("all_qa_pairs", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON格式
    json_file = output_dir / "pmpp_qa_v2.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    # JSONL格式 (Alpaca格式，用于微调)
    jsonl_file = output_dir / "pmpp_qa_v2.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for qa in all_qa:
            item = {
                "instruction": qa.get("question", ""),
                "input": "",
                "output": qa.get("answer", ""),
                "difficulty": qa.get("difficulty", "medium"),
                "topic": qa.get("topic", "CUDA/GPU"),
                "chapter": qa.get("chapter", 0)
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # dataset_info.json (LLaMA-Factory格式)
    info_file = output_dir / "dataset_info.json"
    info = {
        "pmpp_qa_v2": {
            "file_name": "pmpp_qa_v2.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n数据集已保存:")
    print(f"  - JSON: {json_file}")
    print(f"  - JSONL: {jsonl_file}")
    print(f"  - Info: {info_file}")
    print(f"  - 总计: {len(all_qa)} 条问答对")


def load_progress(output_dir: Path) -> Dict:
    """加载进度"""
    progress_file = output_dir / "generation_progress_v2.json"
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "chapter_progress": {},
        "total_generated": 0,
        "all_qa_pairs": [],
        "last_update": None
    }


def save_progress(progress: Dict, output_dir: Path):
    """保存进度"""
    progress["last_update"] = datetime.now().isoformat()
    progress_file = output_dir / "generation_progress_v2.json"
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def print_stats(progress: Dict):
    """打印统计"""
    print("\n" + "="*60)
    print("生成统计")
    print("="*60)

    total = 0
    for ch in sorted(CHAPTERS.keys()):
        ch_key = str(ch)
        ch_prog = progress.get("chapter_progress", {}).get(ch_key, {})
        count = ch_prog.get("generated", 0)
        target = CHAPTERS[ch]["target"]
        title = CHAPTERS[ch]["title"]
        status = "done" if count >= target else "..."
        print(f"  Ch{ch:2d} {title:30s}: {count:4d}/{target:4d} [{status}]")
        total += count

    print(f"\n  总计: {total} 条")
    print("="*60)


def main():
    """主函数"""
    print("="*60)
    print("PMPP问答对生成器 v2")
    print("改进: Few-shot + 质量过滤 + 智能分块")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载进度
    progress = load_progress(OUTPUT_DIR)
    if progress["total_generated"] > 0:
        print(f"\n[续传] 已有进度: {progress['total_generated']}条")
        print_stats(progress)

    # 初始化
    print(f"\n[1/3] 初始化 (模型: {get_model()})...")
    client = init_client()

    print("[2/3] 加载教材...")
    lines = load_textbook()
    print(f"  总行数: {len(lines)}")

    print("[3/3] 开始生成...")

    # 按章节生成
    for chapter_num in sorted(CHAPTERS.keys()):
        try:
            generate_chapter_qa(client, lines, chapter_num, progress)
            save_progress(progress, OUTPUT_DIR)
        except KeyboardInterrupt:
            print("\n\n[中断] 保存进度...")
            save_progress(progress, OUTPUT_DIR)
            print_stats(progress)
            return
        except Exception as e:
            print(f"\n[错误] 第{chapter_num}章: {e}")
            save_progress(progress, OUTPUT_DIR)
            continue

    # 保存最终数据集
    save_dataset(progress, OUTPUT_DIR)
    print_stats(progress)

    print("\n" + "="*60)
    print("生成完成!")
    print("="*60)


if __name__ == "__main__":
    main()
