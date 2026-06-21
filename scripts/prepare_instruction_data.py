#!/usr/bin/env python3
"""
通用中文指令数据集预处理脚本

将 JSON Lines 格式的指令数据集（每行一个 JSON 对象，包含
`instruction`、`input`、`output` 三个字段）转换为训练用的
uint16 二进制 token 文件。

兼容的数据集格式：
  - BELLE        (https://github.com/LianjiaTech/BELLE)
  - Alpaca-CN    (https://github.com/ymcui/Chinese-LLaMA-Alpaca)
  - COIG         (https://huggingface.co/datasets/BAAI/COIG)
  - MOSS-SFT     (https://github.com/OpenLMLab/MOSS)
  - 其他遵循 instruction / input / output 字段约定的中文 SFT 数据集

用法:
  # 默认参数（处理 BELLE 0.5M）
  python scripts/prepare_instruction_data.py

  # 自定义数据集和输出路径
  python scripts/prepare_instruction_data.py \
    --input data/raw/your_dataset.json \
    --output_dir data/processed_your \
    --tokenizer_path checkpoints/your_tokenizer
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanolm.tokenizer import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="通用中文指令数据集预处理")
    parser.add_argument(
        "--input", default="data/raw/Belle_open_source_0.5M.json",
        help="原始 JSONL 数据集路径（默认 BELLE 0.5M）"
    )
    parser.add_argument(
        "--output_dir", default="data/processed_0.5M",
        help="预处理后二进制文件的输出目录"
    )
    parser.add_argument(
        "--tokenizer_path", default="checkpoints/tokenizer_0.5M",
        help="已训练好的分词器路径"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.01,
        help="验证集比例（0.01 = 1%%，大数据集下足够）"
    )
    parser.add_argument(
        "--total_lines", type=int, default=500000,
        help="数据集总行数，仅用于显示进度条"
    )
    return parser.parse_args()


def process_instruction_data(args):
    """
    将 JSONL 指令数据集转换为训练用的 token 二进制文件。

    每条样本按训练模板拼接：
        指令: {instruction}
        [输入: {input}]    # 如果有 input 字段
        回答: {output}

    然后用分词器编码（自动加 BOS/EOS），写入 train.bin / val.bin。
    """
    os.makedirs(args.output_dir, exist_ok=True)
    enc = load_tokenizer(args.tokenizer_path)

    print(f"🚀 开始流式预处理: {args.input}")
    print(f"📍 输出目录: {args.output_dir}")
    print(f"🔤 分词器: {args.tokenizer_path}")

    train_ids, val_ids = [], []

    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=args.total_lines, desc="处理中")):
            try:
                data = json.loads(line)
                instruction = data.get("instruction", "")
                input_str = data.get("input", "")
                output = data.get("output", "")

                # 标准 SFT 模板（与训练时的提示词工程一致）
                text = f"指令: {instruction}\n"
                if input_str:
                    text += f"输入: {input_str}\n"
                text += f"回答: {output}\n"

                # encode 默认 add_bos=True, add_eos=True
                # 每条样本开头是 BOS(2)，结尾是 EOS(3)
                tokens = enc.encode(text)

                # 按比例分配到训练集 / 验证集
                if i % int(1 / args.val_ratio) == 0:
                    val_ids.extend(tokens)
                else:
                    train_ids.extend(tokens)
            except Exception:
                continue

    print("\n💾 正在保存二进制文件...")
    np.array(train_ids, dtype=np.uint16).tofile(
        os.path.join(args.output_dir, 'train.bin')
    )
    np.array(val_ids, dtype=np.uint16).tofile(
        os.path.join(args.output_dir, 'val.bin')
    )

    train_size_mb = len(train_ids) * 2 / 1024**2
    val_size_mb = len(val_ids) * 2 / 1024**2
    print(f"✅ 预处理完成！")
    print(f"   训练集: {len(train_ids):,} tokens ({train_size_mb:.1f} MB)")
    print(f"   验证集: {len(val_ids):,} tokens ({val_size_mb:.1f} MB)")


if __name__ == "__main__":
    args = parse_args()
    process_instruction_data(args)
