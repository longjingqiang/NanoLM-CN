# NanoLM-CN

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A lightweight Chinese language model training framework for personal GPUs. Dataset-agnostic — train on any Chinese corpus. Reference pipeline included for the [BELLE](https://github.com/LianjiaTech/BELLE) instruction dataset.

> 一个面向个人开发者的轻量级中文语言模型训练框架，支持在消费级显卡（RTX 3060 6GB）上从头训练中文模型。框架与数据集解耦，仓库内置 BELLE 数据集的参考流程作为示例，也可适配任何中文文本/指令数据。

---

## English Overview

**NanoLM-CN** is a from-scratch decoder-only Transformer training framework for Chinese, designed for minimal GPU requirements. The framework itself is dataset-agnostic; the included pretrained checkpoint demonstrates training on the BELLE 0.5M Chinese instruction dataset.

**Architecture highlights:**
- RoPE positional encoding + RMSNorm (Pre-Norm) + SwiGLU FFN
- Flash Attention (PyTorch 2.0+ `scaled_dot_product_attention`)
- BPE tokenizer trained on Chinese text (vocab size 8000)
- Weight tying between input embedding and LM head

**Pretrained checkpoint** (`small`, ~37M params, trained on BELLE 0.5M):
- Step 20000, best validation loss: **1.446**
- Hardware: RTX 5060 Laptop GPU (8GB), training time ~6 hours

**Quick start (Windows):**
```bash
pip install -r requirements.txt
# Set UTF-8 output on Windows
$env:PYTHONIOENCODING="utf-8"
# Interactive chat
python scripts/generate.py --interactive
# Or verify the pipeline without any data
python scripts/quickstart.py
```

**Model sizes:**
| Preset | Params | Layers | Dim | VRAM (train) |
|--------|--------|--------|-----|--------------|
| nano   | ~10M   | 6      | 256 | ~2.5 GB      |
| small  | ~37M   | 8      | 512 | ~4.0 GB      |
| medium | ~117M  | 12     | 768 | ~5.5 GB      |
| large  | ~350M  | 16     |1024 | ~11 GB       |

---

## 中文详细说明

### 项目特点

| 特性 | 说明 |
|------|------|
| 🖥️ **资源友好** | 6GB 显存即可训练，支持梯度检查点和混合精度 |
| 🇨🇳 **中文优化** | 专门设计的中文 BPE 分词器，CJK 完整覆盖 |
| ⚡ **现代架构** | RoPE + RMSNorm + SwiGLU + Flash Attention |
| 🔧 **易于定制** | 清晰的代码结构，方便修改模型架构和训练流程 |
| 📊 **完整工具链** | 数据处理 → 训练 → 评估 → 生成，一套完整流程 |

---

### 硬件要求

| 配件 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA 6GB 显存（RTX 3060） | RTX 3070/4060 8GB+ |
| 内存 | 16GB | 32GB |
| 存储 | 20GB | 50GB SSD |

---

### 快速开始

#### 1. 安装

```bash
git clone https://github.com/yourusername/NanoLM-CN.git
cd NanoLM-CN

# 安装 PyTorch（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

> **Windows 用户：** 运行前设置 `$env:PYTHONIOENCODING="utf-8"` 以正确显示中文和 emoji。

#### 2. 验证安装（无需真实数据）

```bash
python scripts/quickstart.py
```

用内置演示文本训练 300 步，验证整个流程正确运行。

#### 3. 交互对话

```bash
# Windows
$env:PYTHONIOENCODING="utf-8"; python scripts/generate.py --interactive

# Linux/macOS
python scripts/generate.py --interactive
```

---

### 完整训练流程（以 BELLE 数据集为例）

#### 步骤一：下载数据

从 [BELLE GitHub](https://github.com/LianjiaTech/BELLE) 下载 `Belle_open_source_0.5M.json`，放入 `data/raw/`：

```
data/
└── raw/
    └── Belle_open_source_0.5M.json   # 500K 指令-回答对
```

#### 步骤二：训练分词器 + 数据预处理

```bash
# 指令数据集专用（默认 BELLE，可通过参数适配 Alpaca/COIG/MOSS 等）
python scripts/prepare_instruction_data.py

# 自定义数据集和路径
python scripts/prepare_instruction_data.py \
  --input data/raw/your_dataset.json \
  --output_dir data/processed_your \
  --tokenizer_path checkpoints/your_tokenizer

# 通用文本预处理（支持 .txt / .jsonl 纯文本预训练）
python scripts/prepare_data.py --input data/raw/*.txt
```

完成后生成：
```
data/processed_belle/
├── train.bin    # 训练集 token 二进制（~184MB）
└── val.bin      # 验证集 token 二进制（~1.9MB）

checkpoints/tokenizer/
└── tokenizer.json
```

#### 步骤三：训练

```bash
# 6GB 显存安全配置（nano 模型）
python scripts/train.py \
  --model nano \
  --batch_size 8 \
  --grad_accum 8 \
  --use_grad_checkpoint

# 8GB 显存配置（small 模型，本项目使用此配置）
python scripts/train.py \
  --model small \
  --batch_size 64 \
  --grad_accum 2 \
  --tokenizer_path checkpoints/tokenizer_0.5M \
  --data_dir data/processed_0.5M

# 从检查点恢复
python scripts/train.py --resume checkpoints/best.pt \
  --tokenizer_path checkpoints/tokenizer_0.5M \
  --data_dir data/processed_0.5M
```

**训练参数参考：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | nano | nano / small / medium / large |
| `--batch_size` | 16 | 每步 batch 大小 |
| `--grad_accum` | 4 | 梯度累积（有效 batch = batch × accum）|
| `--seq_len` | 512 | 序列长度 |
| `--lr` | 自动 | 根据模型大小自动选择（nano:3e-4, small:1e-4）|
| `--max_iters` | 50000 | 总训练步数 |
| `--amp_dtype` | bf16 | 混合精度（旧 GPU 用 fp16）|

#### 步骤四：推理测试

训练完成（或从已有检查点）后，有三种方式与模型交互：

**方式 1：交互对话模式（推荐）**

```bash
# Windows
$env:PYTHONIOENCODING="utf-8"; python scripts/generate.py --interactive

# Linux/macOS
python scripts/generate.py --interactive
```

启动后输入问题即可，脚本会自动把输入包装为训练时的 `指令: ...\n回答: ` 格式：

```
💬 问题: 请介绍一下深度学习
────────────────────────────────────────
🤖 回答:
深度学习是一种人工智能技术，通过多层神经网络模拟人脑的结构...
────────────────────────────────────────
```

**交互模式下的可用命令：**

| 命令 | 作用 |
|------|------|
| `/set temp <值>` | 调整温度（0.7=保守，1.0=默认，1.2=有创意）|
| `/set topk <值>` | 调整 Top-K 采样（推荐 30–50）|
| `/set topp <值>` | 调整 Top-P 核采样（推荐 0.9）|
| `q` / `exit` | 退出 |

**方式 2：单次推理（命令行直接传 prompt）**

```bash
# 自由生成（不加指令模板，适合续写）
python scripts/generate.py --prompt "春眠不觉晓" --temperature 0.8

# 按指令格式生成（手动包装，等价于交互模式内部行为）
python scripts/generate.py --prompt $'指令: 解释一下深度学习\n回答: ' --max_tokens 200
```

**方式 3：自定义推理参数**

```bash
python scripts/generate.py --interactive \
  --checkpoint checkpoints/best.pt \
  --tokenizer_path checkpoints/tokenizer_0.5M \
  --temperature 0.7 \
  --top_k 30 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --max_tokens 250
```

**推理参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | `checkpoints/best.pt` | 模型检查点路径 |
| `--tokenizer_path` | `checkpoints/tokenizer_0.5M` | 分词器路径 |
| `--max_tokens` | 200 | 最大生成 token 数 |
| `--temperature` | 1.0 | 温度（越低越保守）|
| `--top_k` | 50 | Top-K 采样 |
| `--top_p` | 0.9 | 核采样阈值 |
| `--repetition_penalty` | 1.1 | 重复惩罚（>1 减少重复）|

> 详细的推理效果评估见 [`docs/inference_test_report.md`](docs/inference_test_report.md)。

#### 步骤五：定量评估

```bash
# 计算困惑度（PPL）和生成速度
python scripts/evaluate.py

# 绘制训练曲线
python scripts/evaluate.py --plot
```

---

### 模型架构

```
输入 Token IDs
     ↓
Token Embedding
     ↓
┌────────────────────┐
│  Transformer Block │ × N 层
│  RMSNorm           │
│  CausalSelfAttn    │  ← RoPE + Flash Attention
│  RMSNorm           │
│  SwiGLU FFN        │
└────────────────────┘
     ↓
RMSNorm → LM Head（与 Embedding 共享权重）
     ↓
下一个 Token 概率分布
```

---

### 提示词格式

模型训练时使用以下格式，推理时需保持一致：

```
指令: {question}
输入: {context}        # 可选
回答: {answer}
```

`scripts/generate.py --interactive` 已自动处理：用户看到 `问题:` 提示，内部自动包装为 `指令: {输入}\n回答: ` 传给模型。

---

### 显存优化

| 显存 | 推荐配置 |
|------|---------|
| <6GB | nano, batch=4, grad_accum=16, --use_grad_checkpoint |
| 6–8GB | nano, batch=8, grad_accum=8, --use_grad_checkpoint |
| 8–12GB | small, batch=8, grad_accum=4 |
| >12GB | small, batch=16, grad_accum=2 |

---

### 推理测试效果（step 20000）

| 类别 | 效果 |
|------|------|
| 知识解释（AI、编程）| ✅ 良好，概念准确 |
| 建议类（健康、学习）| ✅ 良好，结构清晰 |
| 创作类（短诗）| ⚠️ 一般，有诗意但偶有语病 |
| 代码生成 | ⚠️ 一般，框架正确，细节有误 |
| 常识问答 | ❌ 较差，偶有事实错误 |
| 数学推理 / 翻译 | ❌ 较差 |

详细测试结果见 [`docs/inference_test_report.md`](docs/inference_test_report.md)。

---

### 常见问题

**Q: CUDA out of memory**
```bash
python scripts/train.py --batch_size 4 --grad_accum 16 --use_grad_checkpoint
```

**Q: Windows 下乱码**
```powershell
$env:PYTHONIOENCODING="utf-8"
```

**Q: 训练损失不下降**
- 检查学习率（尝试 `--lr 1e-4` 到 `5e-4`）
- 增加预热步数（`--warmup_iters 2000`）

**Q: 生成文本质量差**
- nano 模型建议至少训练 20k 步
- 降低温度（`--temperature 0.7`）

---

### 参考资料

- [nanoGPT](https://github.com/karpathy/nanoGPT) — 主要参考实现
- [BELLE](https://github.com/LianjiaTech/BELLE) — 中文指令数据集
- [RoPE](https://arxiv.org/abs/2104.09864) — 旋转位置编码
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [LLaMA](https://arxiv.org/abs/2302.13971) — RMSNorm, SwiGLU 参考

---

### 许可证 / License

本项目代码采用 **Apache License 2.0**，与 [BELLE](https://github.com/LianjiaTech/BELLE) 上游协议保持一致。
第三方归属信息见 [`NOTICE`](NOTICE) 文件。

This project is licensed under the **Apache License 2.0**, consistent with the upstream BELLE project.
See the [`NOTICE`](NOTICE) file for third-party attributions.

---

### 声明 / Disclaimer

本项目仅用于个人学习和学术研究，**不得用于商业目的**。
训练数据来自 [BELLE](https://github.com/LianjiaTech/BELLE)，该数据集由 OpenAI API 生成，
使用须遵守 OpenAI 的服务条款。

This project is for personal learning and academic research only. **Not for commercial use.**
Training data is sourced from BELLE (generated via OpenAI API);
usage must comply with OpenAI's Terms of Service.
