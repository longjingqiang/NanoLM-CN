#!/usr/bin/env python3
"""
NanoLM 快速启动演示
无需真实数据，用随机数据验证整个训练流程是否正常运行

用法:
  python scripts/quickstart.py

这会:
1. 构建一个 nano 模型
2. 用合成数据跑 200 步训练
3. 展示生成效果（随机初始化，结果无意义，但验证流程正确）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nanolm.model import NanoLM, PRESET_CONFIGS
from nanolm.tokenizer import CharTokenizer
from nanolm.trainer import Trainer, TrainConfig
from nanolm.utils import get_device, set_seed, print_training_config_summary


# ─── 合成数据集 ───────────────────────────────────────────────────────────────

DEMO_TEXTS = [
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "人工智能是计算机科学的一个重要分支，研究如何让机器像人类一样思考和学习。",
    "深度学习是机器学习的一个子领域，通过多层神经网络来学习数据的表示和特征。",
    "语言模型是一种统计模型，用于预测给定上下文中下一个词的概率分布。",
    "自然语言处理是人工智能和语言学领域的交叉学科，研究计算机如何理解和生成人类语言。",
    "大型语言模型通过在大量文本数据上进行预训练，学习到丰富的语言知识和常识推理能力。",
    "Transformer 架构是现代语言模型的基础，它通过自注意力机制来捕捉序列中的长距离依赖关系。",
    "在有限的计算资源下训练高质量的语言模型是当前研究的重要方向之一。",
    "中文是世界上使用人数最多的语言之一，具有丰富的语言现象和文化内涵。",
    "北京是中国的首都，拥有悠久的历史文化，是政治、文化、国际交往的中心城市。",
    "上海是中国最大的城市，也是国际金融中心和航运中心，经济发展十分活跃。",
    "机器学习算法通过从数据中学习规律，自动改进系统的性能，而无需明确编程。",
    "神经网络由大量相互连接的节点（神经元）组成，能够学习复杂的非线性映射关系。",
    "梯度下降是训练神经网络最常用的优化算法，通过计算损失函数对参数的梯度来更新参数。",
] * 50  # 重复以增加数据量


class DemoDataset(Dataset):
    """演示用数据集"""

    def __init__(self, tokenizer, seq_len: int = 128, n_samples: int = 2000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Tokenize 所有文本
        all_ids = [tokenizer.bos_id]
        for text in DEMO_TEXTS:
            all_ids.extend(tokenizer.encode(text, add_bos=False, add_eos=True))

        total = torch.tensor(all_ids, dtype=torch.long)

        # 生成样本
        self.samples = []
        for i in range(0, min(n_samples * seq_len, len(total) - seq_len - 1), seq_len):
            self.samples.append(total[i:i + seq_len + 1])
            if len(self.samples) >= n_samples:
                break

        print(f"  演示数据集: {len(self.samples)} 个样本, {len(all_ids):,} tokens")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        return chunk[:-1], chunk[1:]


def main():
    print("=" * 60)
    print("🚀 NanoLM 快速启动演示")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  未检测到 GPU，使用 CPU 运行（速度较慢，建议用 GPU）")

    # 构建迷你分词器
    print("\n步骤 1: 构建演示分词器...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(DEMO_TEXTS, min_freq=1, max_vocab=2000)
    print(f"  词表大小: {tokenizer.vocab_size}")

    # 构建 nano 模型（最小配置）
    print("\n步骤 2: 构建模型...")
    model_config = PRESET_CONFIGS["nano"]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = 128
    model_config.n_layers = 4     # 减小以加速演示
    model_config.n_embd = 128
    model_config.n_heads = 4

    model = NanoLM(model_config).to(device)

    # 数据集
    print("\n步骤 3: 准备演示数据...")
    seq_len = 128
    train_dataset = DemoDataset(tokenizer, seq_len=seq_len, n_samples=2000)
    val_dataset   = DemoDataset(tokenizer, seq_len=seq_len, n_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, drop_last=True)

    # 训练配置
    train_config = TrainConfig(
        max_iters=300,
        batch_size=16,
        grad_accum=1,
        seq_len=seq_len,
        lr=5e-4,
        lr_min=5e-5,
        warmup_iters=50,
        lr_decay_iters=250,
        use_amp=torch.cuda.is_available(),
        use_grad_checkpoint=False,  # 演示不需要
        log_interval=50,
        eval_interval=100,
        save_interval=9999,         # 演示不保存
        output_dir="/tmp/nanolm_demo",
    )

    print_training_config_summary(model, train_config)

    # 训练
    print("\n步骤 4: 开始训练 (300 步演示)...")
    trainer = Trainer(model, train_loader, val_loader, train_config, device)
    trainer.train()

    # 演示生成
    print("\n步骤 5: 文本生成演示...")
    model.eval()

    test_prompts = [
        "人工智能",
        "春眠",
        "深度学习",
    ]

    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model.generate(
                x, max_new_tokens=50, temperature=0.8, top_k=20
            )

        generated = tokenizer.decode(output[0, len(input_ids):].tolist())
        print(f"\n  提示词: {prompt}")
        print(f"  生成:   {generated[:80]}")

    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print()
    print("📌 整个训练流程验证通过。")
    print("   若要训练真实的中文语言模型，请按以下步骤操作:")
    print()
    print("1️⃣  准备中文语料 (txt 格式，每行一段文本)")
    print("   推荐数据集: 维基百科中文、新闻语料、古诗词等")
    print()
    print("2️⃣  运行数据预处理:")
    print("   python scripts/prepare_data.py --input data/raw/*.txt")
    print()
    print("3️⃣  开始训练:")
    print("   python scripts/train.py --model nano")
    print()
    print("4️⃣  生成文本:")
    print("   python scripts/generate.py --interactive")
    print("=" * 60)


if __name__ == "__main__":
    main()
