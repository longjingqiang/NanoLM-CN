#!/usr/bin/env python3
"""
NanoLM 模型评估脚本

计算以下指标:
- 困惑度 (Perplexity) - 越低越好
- 每秒生成 token 数 (吞吐量)
- 显存占用

用法:
  python scripts/evaluate.py
  python scripts/evaluate.py --checkpoint checkpoints/best.pt
  python scripts/evaluate.py --plot  # 绘制训练曲线
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import time
import json
import torch
import numpy as np
from torch.cuda.amp import autocast

from nanolm.model import NanoLM, PRESET_CONFIGS
from nanolm.tokenizer import load_tokenizer
from nanolm.dataset import MemmapDataset, create_dataloader
from nanolm.utils import get_device
from nanolm.trainer import get_gpu_memory_info


def parse_args():
    parser = argparse.ArgumentParser(description="NanoLM 模型评估")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--tokenizer_path", default="checkpoints/tokenizer_0.5M")
    parser.add_argument("--data_dir", default="data/processed_0.5M")
    parser.add_argument("--model", default="small")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_iters", type=int, default=200,
                        help="评估步数（越多越准确）")
    parser.add_argument("--plot", action="store_true",
                        help="绘制训练曲线")
    parser.add_argument("--log_path", default="checkpoints/training_log.json",
                        help="训练日志路径（用于绘图）")
    return parser.parse_args()


@torch.no_grad()
def compute_perplexity(model, dataloader, device, eval_iters: int, use_amp: bool = True):
    """计算困惑度"""
    model.eval()
    losses = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    data_iter = iter(dataloader)

    for i in range(eval_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                      dtype=amp_dtype, enabled=use_amp and torch.cuda.is_available()):
            _, loss = model(x, y)
        losses.append(loss.item())

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{eval_iters}, 当前损失: {sum(losses)/len(losses):.4f}")

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def benchmark_generation(model, tokenizer, device, n_tokens: int = 200):
    """测试生成速度"""
    prompt = "人工智能是"
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 预热
    with torch.no_grad():
        model.generate(x, max_new_tokens=10, temperature=1.0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output = model.generate(x, max_new_tokens=n_tokens, temperature=0.8)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    generated = output[0, len(input_ids):]
    tokens_per_sec = len(generated) / (t1 - t0)
    return tokens_per_sec, tokenizer.decode(generated.tolist())


def main():
    args = parse_args()
    device = get_device()

    # 绘制训练曲线
    if args.plot:
        from nanolm.utils import plot_training_curve
        if os.path.exists(args.log_path):
            plot_training_curve(args.log_path,
                                save_path=args.log_path.replace(".json", ".png"))
        else:
            print(f"❌ 找不到训练日志: {args.log_path}")
        return

    # 加载分词器
    tokenizer = load_tokenizer(args.tokenizer_path)

    # 构建模型
    model_config = PRESET_CONFIGS[args.model]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = args.seq_len
    model = NanoLM(model_config).to(device)

    # 加载检查点
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict, strict=False)
        step = ckpt.get("step", "?")
        print(f"✅ 已加载检查点 (step={step})")
    else:
        print(f"⚠️  使用随机初始化模型（找不到: {args.checkpoint}）")

    model.eval()

    print(f"\n{'='*55}")
    print(f"  NanoLM 评估报告")
    print(f"{'='*55}")

    # 模型信息
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 模型信息:")
    print(f"  参数量: {n_params/1e6:.2f}M")
    print(f"  模型大小: {n_params*4/1024**2:.1f} MB (fp32)")
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  最大序列长度: {args.seq_len}")

    # 困惑度评估
    val_bin = os.path.join(args.data_dir, "val.bin")
    if os.path.exists(val_bin):
        print(f"\n📐 计算困惑度 (验证集)...")
        val_dataset = MemmapDataset(val_bin, args.seq_len, "val")
        val_loader = create_dataloader(
            val_dataset, args.batch_size, shuffle=False, num_workers=0
        )
        avg_loss, ppl = compute_perplexity(
            model, val_loader, device, args.eval_iters
        )
        print(f"\n  ✅ 平均损失: {avg_loss:.4f}")
        print(f"  ✅ 困惑度 (PPL): {ppl:.2f}")

        # 解读困惑度
        if ppl < 50:
            quality = "🌟 优秀 - 模型已学到较多语言规律"
        elif ppl < 100:
            quality = "👍 良好 - 模型有一定语言理解"
        elif ppl < 200:
            quality = "📈 一般 - 模型仍在学习中"
        else:
            quality = "🔰 较差 - 可能训练步数不足或数据质量待提升"
        print(f"  质量评估: {quality}")
    else:
        print(f"\n⚠️  找不到验证数据: {val_bin}，跳过困惑度计算")

    # 生成速度测试
    print(f"\n⚡ 生成速度测试...")
    try:
        tokens_per_sec, sample = benchmark_generation(model, tokenizer, device)
        print(f"  生成速度: {tokens_per_sec:.1f} tokens/秒")
        chars_per_sec = tokens_per_sec  # 大约等于字符数
        print(f"  约 {chars_per_sec:.0f} 字/秒")
        print(f"\n  示例生成文本:")
        print(f"  「人工智能是{sample[:100]}」")
    except Exception as e:
        print(f"  ⚠️  速度测试失败: {e}")

    # 显存报告
    if torch.cuda.is_available():
        mem = get_gpu_memory_info()
        print(f"\n💾 显存占用:")
        print(f"  当前: {mem['allocated']:.2f} GB")
        print(f"  峰值: {mem['max_allocated']:.2f} GB")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
