#!/usr/bin/env python3
"""
NanoLM 主训练脚本

用法:
  # 使用默认配置（nano 模型）
  python scripts/train.py

  # 指定模型大小
  python scripts/train.py --model small --batch_size 8

  # 从检查点恢复
  python scripts/train.py --resume checkpoints/step_0010000.pt

  # 快速测试（小步数）
  python scripts/train.py --max_iters 100 --eval_interval 50

  # 针对 6GB 显存优化
  python scripts/train.py --model nano --batch_size 8 --grad_accum 8 --use_grad_checkpoint
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from nanolm.model import NanoLM, PRESET_CONFIGS
from nanolm.tokenizer import load_tokenizer
from nanolm.dataset import MemmapDataset, TextDataset, create_dataloader
from nanolm.trainer import Trainer, TrainConfig
from nanolm.utils import set_seed, get_device, print_training_config_summary


def parse_args():
    parser = argparse.ArgumentParser(description="NanoLM 训练")

    # 模型
    parser.add_argument("--model", default="nano",
                        choices=["nano", "small", "medium", "large"],
                        help="模型预设大小")

    # 数据
    parser.add_argument("--data_dir", default="data/processed",
                        help="预处理数据目录（包含 train.bin, val.bin）")
    parser.add_argument("--tokenizer_path", default="checkpoints/tokenizer",
                        help="分词器路径")
    parser.add_argument("--output_dir", default="checkpoints",
                        help="检查点保存目录")

    # 训练超参数
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None,
                        help="峰值学习率（默认按模型大小自动设置）")
    parser.add_argument("--warmup_iters", type=int, default=None,
                        help="预热步数（默认按模型大小自动设置）")

    # 显存优化
    parser.add_argument("--use_grad_checkpoint", action="store_true", default=True,
                        help="梯度检查点（节省显存）")
    parser.add_argument("--no_grad_checkpoint", dest="use_grad_checkpoint",
                        action="store_false")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"],
                        help="混合精度类型（Ampere GPU 用 bf16，旧 GPU 用 fp16）")
    parser.add_argument("--compile", action="store_true",
                        help="使用 torch.compile 加速（需要 PyTorch 2.0+）")

    # 日志
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=2000)

    # 恢复训练
    parser.add_argument("--resume", default=None,
                        help="从检查点文件恢复训练")

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check_data_files(data_dir: str) -> bool:
    """检查数据文件是否存在"""
    train_path = os.path.join(data_dir, "train.bin")
    val_path   = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path):
        print(f"❌ 找不到训练数据: {train_path}")
        print("  请先运行数据预处理脚本:")
        print("  python scripts/prepare_data.py --input your_data/*.txt")
        return False

    if not os.path.exists(val_path):
        print(f"❌ 找不到验证数据: {val_path}")
        return False

    train_mb = os.path.getsize(train_path) / 1024**2
    val_mb   = os.path.getsize(val_path) / 1024**2
    print(f"✅ 数据文件: train={train_mb:.1f}MB, val={val_mb:.1f}MB")
    return True


def estimate_recommended_config(vram_gb: float, args) -> dict:
    """根据显存大小推荐配置"""
    recs = {}
    if vram_gb < 6:
        recs["model"] = "nano"
        recs["batch_size"] = 4
        recs["grad_accum"] = 16
        recs["seq_len"] = 256
        recs["use_grad_checkpoint"] = True
    elif vram_gb < 8:
        recs["model"] = "nano"
        recs["batch_size"] = 8
        recs["grad_accum"] = 8
        recs["seq_len"] = 512
        recs["use_grad_checkpoint"] = True
    elif vram_gb < 12:
        recs["model"] = "small"
        recs["batch_size"] = 8
        recs["grad_accum"] = 4
        recs["seq_len"] = 512
    else:
        recs["model"] = "small"
        recs["batch_size"] = 16
        recs["grad_accum"] = 2
        recs["seq_len"] = 512
    return recs


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # 根据显存给出建议
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 8:
            recs = estimate_recommended_config(vram_gb, args)
            print(f"\n💡 显存建议配置 ({vram_gb:.1f}GB):")
            for k, v in recs.items():
                print(f"   {k}: {v}")
            print(f"   (当前使用: model={args.model}, batch={args.batch_size})")

    # 检查数据文件
    if not check_data_files(args.data_dir):
        sys.exit(1)

    # 加载分词器
    print(f"\n加载分词器: {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path)

    # 构建模型
    print(f"\n构建模型: {args.model}")
    model_config = PRESET_CONFIGS[args.model]
    model_config.vocab_size = tokenizer.vocab_size  # 同步词表大小
    model_config.max_seq_len = args.seq_len

    model = NanoLM(model_config).to(device)

    # 每个模型尺寸的推荐学习率（参数越多，LR 越保守）
    MODEL_LR_DEFAULTS = {
        "nano":   (3e-4, 3e-5,  800),   # (peak_lr, min_lr, warmup)
        "small":  (1e-4, 1e-5, 2000),
        "medium": (6e-5, 6e-6, 3000),
        "large":  (3e-5, 3e-6, 4000),
    }
    peak_lr, min_lr, warmup = MODEL_LR_DEFAULTS[args.model]
    if args.lr is not None:
        peak_lr = args.lr
    if args.warmup_iters is not None:
        warmup = args.warmup_iters

    print(f"\n📐 学习率配置: peak={peak_lr:.1e}, min={min_lr:.1e}, warmup={warmup} 步")

    # 训练配置
    train_config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        model_preset=args.model,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        lr=peak_lr,
        lr_min=min_lr,
        warmup_iters=warmup,
        use_grad_checkpoint=args.use_grad_checkpoint,
        amp_dtype=args.amp_dtype,
        compile_model=args.compile,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )

    # 打印配置摘要
    print_training_config_summary(model, train_config)

    # 数据集
    num_workers = 0 if sys.platform == "win32" else train_config.num_workers
    train_dataset = MemmapDataset(
        os.path.join(args.data_dir, "train.bin"), args.seq_len, "train"
    )
    val_dataset = MemmapDataset(
        os.path.join(args.data_dir, "val.bin"), args.seq_len, "val"
    )
    train_loader = create_dataloader(
        train_dataset, args.batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = create_dataloader(
        val_dataset, args.batch_size, shuffle=False, num_workers=num_workers
    )

    # 训练器
    trainer = Trainer(model, train_loader, val_loader, train_config, device)

    # 恢复检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 保存训练配置
    train_config.save(os.path.join(args.output_dir, "train_config.json"))

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⏹️  训练被用户中断")
        save_path = trainer.save_checkpoint("interrupted")
        print(f"已保存当前进度: {save_path}")

    print("\n✅ 完成！")
    print(f"最佳模型: {args.output_dir}/best.pt")
    print(f"最终模型: {args.output_dir}/final.pt")
    print(f"\n下一步:")
    print(f"  生成文本: python scripts/generate.py --prompt '你好'")
    print(f"  评估模型: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
