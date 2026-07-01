#!/usr/bin/env python3
"""
数据预处理脚本

用法:
  # 从文本文件构建分词器
  python scripts/prepare_data.py --mode tokenizer --input data/raw/*.txt

  # 将文本转换为 token 二进制文件
  python scripts/prepare_data.py --mode tokenize --input data/raw/*.txt

  # 一键完成（构建词表 + tokenize）
  python scripts/prepare_data.py --mode all --input data/raw/*.txt

支持的数据格式:
  - 纯文本文件 (.txt)，每行一段文本
  - JSON Lines (.jsonl)，包含 "text" 字段
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
from pathlib import Path
from nanolm.tokenizer import (
    CharTokenizer, BPETokenizer, build_tokenizer_from_files
)
from nanolm.dataset import prepare_binary_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="NanoLM 数据预处理")
    parser.add_argument(
        "--mode", choices=["tokenizer", "tokenize", "all"], default="all",
        help="处理模式: tokenizer(仅构建词表), tokenize(仅转换), all(全部)"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="输入文件路径（支持通配符，如 data/raw/*.txt）"
    )
    parser.add_argument(
        "--output_dir", default="data/processed",
        help="输出目录"
    )
    parser.add_argument(
        "--tokenizer_dir", default="checkpoints/tokenizer",
        help="分词器保存目录"
    )
    parser.add_argument(
        "--tokenizer_type", choices=["bpe", "char"], default="bpe",
        help="分词器类型: bpe（推荐）或 char（字符级）"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=8000,
        help="词表大小（建议 6000~16000）"
    )
    parser.add_argument(
        "--min_freq", type=int, default=2,
        help="最小词频（低于此频率的字符不加入词表）"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.95,
        help="训练集比例（剩余为验证集）"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="最大样本数限制（用于快速测试）"
    )
    return parser.parse_args()


def expand_glob(patterns):
    """展开通配符文件路径"""
    files = []
    for pattern in patterns:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            files.extend(expanded)
        elif os.path.exists(pattern):
            files.append(pattern)
        else:
            print(f"⚠️  找不到文件: {pattern}")
    return sorted(set(files))


def read_texts_from_files(files):
    """从文件列表读取文本行"""
    texts = []
    for f in files:
        ext = Path(f).suffix.lower()
        print(f"读取: {f}")
        if ext == ".jsonl" or ext == ".json":
            import json
            with open(f, encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    try:
                        item = json.loads(line.strip())
                        text = item.get("text", item.get("content", ""))
                        if text and len(text) > 10:
                            texts.append(text)
                    except:
                        continue
        else:
            # 纯文本
            with open(f, encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) > 10:
                        texts.append(line)
    return texts


def main():
    args = parse_args()

    # 展开文件路径
    input_files = expand_glob(args.input)
    if not input_files:
        print("❌ 错误: 未找到任何输入文件")
        print("  请确认路径正确，例如: python scripts/prepare_data.py --input data/raw/test.txt")
        sys.exit(1)

    print(f"\n📁 找到 {len(input_files)} 个输入文件:")
    for f in input_files[:5]:
        size_mb = os.path.getsize(f) / 1024**2
        print(f"  {f} ({size_mb:.1f} MB)")
    if len(input_files) > 5:
        print(f"  ... 共 {len(input_files)} 个文件")

    # 构建分词器
    if args.mode in ("tokenizer", "all"):
        print(f"\n{'='*50}")
        print(f"步骤 1: 构建 {args.tokenizer_type.upper()} 分词器")
        print(f"{'='*50}")

        texts = read_texts_from_files(input_files)
        if not texts:
            print("❌ 错误: 未读取到任何文本")
            sys.exit(1)

        print(f"共读取 {len(texts):,} 行文本")

        if args.tokenizer_type == "bpe":
            tokenizer = BPETokenizer()
            tokenizer.train(texts, vocab_size=args.vocab_size, min_freq=args.min_freq)
        else:
            tokenizer = CharTokenizer()
            tokenizer.build_vocab(texts, min_freq=args.min_freq, max_vocab=args.vocab_size)

        tokenizer.save(args.tokenizer_dir)

        # 测试分词
        test_text = "人工智能是计算机科学的一个重要分支，研究如何让机器像人类一样思考。"
        ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(ids)
        print(f"\n🔤 分词测试:")
        print(f"  原文: {test_text}")
        print(f"  Token 数: {len(ids)}")
        print(f"  还原: {decoded}")
    else:
        # 加载已有分词器
        from nanolm.tokenizer import load_tokenizer
        print(f"\n加载分词器: {args.tokenizer_dir}")
        tokenizer = load_tokenizer(args.tokenizer_dir)

    # Tokenize 数据
    if args.mode in ("tokenize", "all"):
        print(f"\n{'='*50}")
        print(f"步骤 2: Tokenize 数据集")
        print(f"{'='*50}")

        os.makedirs(args.output_dir, exist_ok=True)
        prepare_binary_dataset(
            input_files=input_files,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            train_ratio=args.train_ratio,
        )

    print(f"\n✅ 数据预处理完成！")
    print(f"   分词器: {args.tokenizer_dir}")
    if args.mode in ("tokenize", "all"):
        print(f"   数据:   {args.output_dir}/train.bin, {args.output_dir}/val.bin")
    print(f"\n下一步: python scripts/train.py")


if __name__ == "__main__":
    main()
