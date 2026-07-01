#!/usr/bin/env python3
"""
从 HuggingFace Hub 下载中文指令数据集到 data/raw/

默认下载 BELLE 0.5M 作为示例；也可通过命令行参数下载任意
HuggingFace 数据集。国内用户默认走 hf-mirror.com 镜像加速。

用法:
  # 下载默认数据集（BELLE 0.5M）
  python scripts/download_data.py

  # 下载其他数据集
  python scripts/download_data.py \
    --repo_id YeungNLP/firefly-train-1.1M \
    --filename firefly-train-1.1M.jsonl
"""

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="从 HuggingFace 下载指令数据集")
    parser.add_argument(
        "--repo_id", default="BelleGroup/train_0.5M_CN",
        help="HuggingFace 数据集仓库 ID（默认：BELLE 0.5M）"
    )
    parser.add_argument(
        "--filename", default="Belle_open_source_0.5M.json",
        help="要下载的文件名"
    )
    parser.add_argument(
        "--save_dir", default="data/raw",
        help="保存目录（默认 data/raw）"
    )
    parser.add_argument(
        "--mirror", default="https://hf-mirror.com",
        help="HuggingFace 镜像站地址（国内用户加速）；置空则用官方源"
    )
    return parser.parse_args()


def download_dataset(args):
    """下载 HuggingFace Hub 上的单个数据文件。

    必须在 os.environ["HF_ENDPOINT"] 设置之后调用，
    因为 huggingface_hub 在导入时读取该环境变量。
    """
    from huggingface_hub import hf_hub_download

    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, args.save_dir)
    os.makedirs(save_path, exist_ok=True)

    print(f"🚀 开始下载数据集...")
    print(f"📦 仓库: {args.repo_id}")
    print(f"📄 文件: {args.filename}")
    print(f"💾 保存到: {save_path}")
    if args.mirror:
        print(f"🌐 镜像: {args.mirror}")

    try:
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"\n✅ 下载成功！")
        print(f"📍 文件位置: {path}")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("💡 提示: 请确认 repo_id 和 filename 是否正确。")


if __name__ == "__main__":
    args = parse_args()

    # 必须在 import huggingface_hub 之前设置，否则镜像不生效
    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.getcwd(), ".hf_cache")

    download_dataset(args)
