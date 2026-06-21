# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

NanoLM-CN 是一个面向个人开发者的轻量级中文语言模型训练框架，目标是在 RTX 3060（6GB 显存）上从头训练 Decoder-Only Transformer 模型。框架与数据集解耦，仓库内置 BELLE 数据集的参考预处理流程作为示例，可适配任意中文文本/指令数据。

## 常用命令

```bash
# 交互式对话（推荐，已内置提示词工程）
# Windows 需设置编码: $env:PYTHONIOENCODING="utf-8"
python scripts/generate.py --interactive

# 单次推理（使用原始 prompt，不加指令模板）
python scripts/generate.py --prompt "春眠不觉晓" --temperature 0.8 --top_k 40

# 按指令格式推理（手动包装，等价于 --interactive 内部格式）
python scripts/generate.py --prompt $'指令: 解释一下深度学习\n回答: ' --max_tokens 200

# 模型评估（困惑度 + 生成速度）
python scripts/evaluate.py

# 快速验证整个流程（无需真实数据）
python scripts/quickstart.py

# 数据预处理（0.5M BELLE 数据集）
python scripts/prepare_instruction_data.py

# 训练（small 模型，从检查点恢复）
python scripts/train.py --model small --resume checkpoints/best.pt

# 训练（6GB 显存下的 nano 模型）
python scripts/train.py --model nano --batch_size 8 --grad_accum 8 --use_grad_checkpoint
```

## 训练数据格式

BELLE 数据集以以下格式存储，**模型的提示词工程必须与此匹配**：

```
指令: {instruction}
输入: {input}          # 可选，有 input 时才出现
回答: {output}<eos>
```

`scripts/generate.py` 的交互模式已自动处理：用户看到 `问题:` 提示，内部传给模型的是 `"指令: {用户输入}\n回答: "`。

## 架构概览

### 核心模块 (`nanolm/`)

**`model.py`** — 主模型，Decoder-Only Transformer：
- 位置编码：`RotaryEmbedding`（RoPE），相比绝对位置编码外推性更好
- 归一化：`RMSNorm`（Pre-Norm 结构）
- 注意力：`CausalSelfAttention`（多头因果注意力 + PyTorch 2.0 Flash Attention）
- 前馈网络：`FeedForward`（SwiGLU 激活）
- 权重绑定：输入 Embedding 与 LM Head 共享权重
- 支持梯度检查点（`use_grad_checkpoint`）、Top-K/Top-P/重复惩罚采样

模型规格（`ModelConfig` 预设）：
| 预设 | 参数量 | 层数 | 维度 |
|------|--------|------|------|
| nano | ~10M | 6 | 256 |
| small | ~37M | 8 | 512 |
| medium | ~117M | 12 | 768 |
| large | ~350M | 16 | 1024 |

**`tokenizer.py`** — 中文分词器，两种模式：
- `CharTokenizer`：字符级，适合小数据集快速实验，无 OOV 问题
- `BPETokenizer`：字节对编码，适合大数据集，词表大小推荐 6000–16000
- 特殊 token：`<pad>=0, <unk>=1, <bos>=2, <eos>=3, <sep>=4, <mask>=5`
- 中文处理：CJK 基本区 + 扩展 A/B 完整覆盖，全角转半角标准化

**`dataset.py`** — 数据集加载：
- `MemmapDataset`：内存映射 `.bin` 文件，适合大数据，多进程友好
- `TextDataset`：直接从 `.txt/.jsonl` 加载，整个数据集入内存
- `ChatDataset`：对话指令微调（SFT）格式，支持 instruction-output 和 messages 格式

**`trainer.py`** — 训练引擎：
- Cosine 学习率调度 + 线性预热（`warmup_iters`）
- AdamW 优化器，分组权重衰减（Embedding 和 Bias 不衰减）
- 混合精度（fp16/bf16，`use_amp=True`）
- 梯度累积（`grad_accum`）+ 梯度裁剪
- 自动保留最近 N 个检查点（`keep_checkpoints=3`）+ 早停

### 数据流

```
原始文本 (.txt / .json)
  → prepare_instruction_data.py    # 构建分词器 + tokenize + 切分 train/val
  → data/processed_0.5M/*.bin    # uint16 内存映射文件
  → MemmapDataset           # 滑动窗口采样序列
  → train.py                # 训练循环
  → checkpoints/*.pt        # 模型权重
  → generate.py             # 推理
```

### 显存优化组合

| 显存 | 推荐配置 |
|------|---------|
| <6GB | nano, batch=4, grad_accum=16, grad_checkpoint |
| 6–8GB | nano, batch=8, grad_accum=8, grad_checkpoint |
| 8–12GB | small, batch=8, grad_accum=4 |
| >12GB | small, batch=16, grad_accum=2 |

## 已有数据和检查点

- `data/raw/belle_0.5m.txt`：500K 样本（259MB）
- `data/processed_0.5M/train.bin`：约 96M tokens（184.7MB）
- `checkpoints/best.pt`：**主要推理检查点**，small 模型，step 20000，最佳验证损失 1.5558
- `checkpoints/step_0020000.pt`：同上的带步数命名版本
- `checkpoints/tokenizer_0.5M/`：BPE 分词器（vocab_size=8000，对应 0.5M 数据集）
- `checkpoints/train_config.json`：训练参数配置

## 依赖环境

见 `requirements.txt`。需要 PyTorch 2.0+ 以使用 Flash Attention（`F.scaled_dot_product_attention`）。Windows 控制台需要设置 `$env:PYTHONIOENCODING="utf-8"` 以正确显示中文和 emoji。
