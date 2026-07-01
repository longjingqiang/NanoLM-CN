#!/usr/bin/env python3
"""
NanoLM 文本生成脚本

用法:
  # 交互式对话
  python scripts/generate.py --interactive

  # 单次生成
  python scripts/generate.py --prompt "春眠不觉晓"

  # 调整生成参数
  python scripts/generate.py --prompt "人工智能" --max_tokens 200 --temperature 0.8 --top_k 40

  # 使用最佳检查点
  python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "从前有座山"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from nanolm.model import NanoLM, ModelConfig, PRESET_CONFIGS
from nanolm.tokenizer import load_tokenizer
from nanolm.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="NanoLM 文本生成")

    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="模型检查点路径")
    parser.add_argument("--tokenizer_path", default="checkpoints/tokenizer_0.5M",
                        help="分词器路径")
    parser.add_argument("--model", default="small",
                        choices=["nano", "small", "medium", "large"],
                        help="模型大小（如果没有 checkpoint 则用此初始化）")

    parser.add_argument("--prompt", default="",
                        help="生成提示词")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="温度（0.1=保守, 1.0=随机, 1.5=创意）")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K 采样")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-P 核采样")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="重复惩罚（>1 减少重复）")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="生成几条结果")

    parser.add_argument("--interactive", action="store_true",
                        help="交互模式")
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def load_model(checkpoint_path: str, model_preset: str, tokenizer, device: str):
    """加载模型"""
    model_config = PRESET_CONFIGS[model_preset]
    model_config.vocab_size = tokenizer.vocab_size

    if os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # 从检查点恢复模型配置（如果有）
        if "config" in ckpt:
            train_cfg = ckpt["config"]
            preset = train_cfg.get("model_preset", model_preset)
            model_config = PRESET_CONFIGS.get(preset, model_config)
            model_config.vocab_size = tokenizer.vocab_size
            model_config.max_seq_len = train_cfg.get("seq_len", 512)

        model = NanoLM(model_config).to(device)

        # 加载权重（忽略不匹配的键）
        state_dict = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  缺失键: {missing[:3]}...")
        if unexpected:
            print(f"⚠️  多余键: {unexpected[:3]}...")
        print(f"✅ 模型已加载 (步骤 {ckpt.get('step', '?')})")
    else:
        print(f"⚠️  检查点不存在: {checkpoint_path}")
        print(f"   使用随机初始化的 {model_preset} 模型（生成结果为随机）")
        model = NanoLM(model_config).to(device)

    return model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: str,
) -> str:
    """生成文本，遇到 EOS 自动截断（防止串台）"""
    # 编码提示词
    if prompt:
        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    else:
        input_ids = [tokenizer.bos_id]

    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 生成
    with torch.no_grad():
        output = model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_at_eos=True,               # ← 遇 EOS 停止，防止串台
            eos_token_id=tokenizer.eos_id,
            min_new_tokens=5,               # ← 最少生成5个token，避免过早结束
        )

    # 解码新生成的部分（EOS 之后的内容已被截断）
    generated_ids = output[0, len(input_ids):].tolist()
    # 再次双重保险：手动截断到第一个 EOS
    if tokenizer.eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(tokenizer.eos_id)]

    # 解码生成文本
    generated_text = tokenizer.decode(generated_ids)

    # 防止串台：检测生成文本中是否出现新的指令/问题模式，并截断
    # 加入了半角冒号（英文冒号），以严格匹配你的数据集
    stop_patterns = [
        "\n指令:",  # 👈 核心修改：匹配数据集的半角冒号
        "\n指令：", # 兼容全角
        "\n问题:",
        "\n问题：",
        "\n输入:",
        "\n输入：",
    ]

    # 查找第一个停止模式的位置
    earliest_stop = len(generated_text)
    for pattern in stop_patterns:
        idx = generated_text.find(pattern)
        if idx != -1 and idx < earliest_stop:
            earliest_stop = idx

    # 如果找到停止模式，截断到该位置
    if earliest_stop < len(generated_text):
        generated_text = generated_text[:earliest_stop]
        print(f"检测到新的指令/问题，已截断生成文本")

    return generated_text


def interactive_mode(model, tokenizer, args, device):
    """交互模式"""
    print("\n" + "=" * 60)
    print("🤖 NanoLM 交互对话模式")
    print("   输入你的问题开始对话，输入 'q' 退出")
    print("   输入 '/set temp <值>' 调整温度（默认1.0）")
    print("   输入 '/set topk <值>' 调整 top_k（默认50）")
    print("   输入 '/set topp <值>' 调整 top_p（默认0.9）")
    print("-" * 60)
    print("   示例问题: 人工智能是什么？")
    print("   示例问题: 请介绍一下深度学习")
    print("   示例问题: 如何学好编程？")
    print("=" * 60)

    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p

    while True:
        try:
            user_input = input("\n问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit", "退出"):
            print("再见！")
            break

        # 设置命令
        if user_input.startswith("/set"):
            parts = user_input.split()
            if len(parts) == 3:
                key, val = parts[1], parts[2]
                try:
                    if key == "temp":
                        temperature = float(val)
                        print(f"✅ 温度设为 {temperature}")
                    elif key == "topk":
                        top_k = int(val)
                        print(f"✅ Top-K 设为 {top_k}")
                    elif key == "topp":
                        top_p = float(val)
                        print(f"✅ Top-P 设为 {top_p}")
                except ValueError:
                    print("❌ 无效值")
            continue

        
        print(f"回答:")

        # 内部使用训练时的格式 "指令: ...\n回答: " 以匹配训练数据分布
        formatted_prompt = f"指令: {user_input}\n回答: "

        result = generate_text(
            model, tokenizer, formatted_prompt,
            max_tokens=args.max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )

        print(result.strip())
        


def main():
    args = parse_args()

    if args.seed:
        set_seed(args.seed)

    device = get_device()

    # 加载分词器
    tokenizer = load_tokenizer(args.tokenizer_path)

    # 加载模型
    model = load_model(args.checkpoint, args.model, tokenizer, device)
    model.eval()

    if args.interactive:
        interactive_mode(model, tokenizer, args, device)
        return

    # 单次或批量生成
    if not args.prompt:
        args.prompt = input("请输入提示词: ").strip()

    print(f"\n{'='*60}")
    print(f"提示词: {args.prompt}")
    print(f"参数: temperature={args.temperature}, top_k={args.top_k}, "
          f"top_p={args.top_p}, max_tokens={args.max_tokens}")
    print(f"{'='*60}\n")

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"─── 样本 {i+1}/{args.num_samples} ───")

        result = generate_text(
            model, tokenizer, args.prompt,
            args.max_tokens, args.temperature,
            args.top_k, args.top_p,
            args.repetition_penalty, device,
        )

        print(args.prompt + result)
        print()


if __name__ == "__main__":
    main()
