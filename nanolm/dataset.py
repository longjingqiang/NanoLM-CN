"""
NanoLM 数据集
高效的中文文本数据加载和预处理
"""

import os
import struct
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Iterator, Tuple
from pathlib import Path


# ─── 内存映射数据集（推荐，适合大数据）─────────────────────────────────────────

class MemmapDataset(Dataset):
    """
    内存映射 Token 数据集

    将预处理好的 token ids 存储为二进制文件，用 numpy memmap 读取
    - 内存占用极低（不需要把整个数据集加载到内存）
    - 随机访问速度快
    - 支持多进程 DataLoader

    使用前需先运行 prepare_data.py 生成 .bin 文件
    """

    def __init__(self, bin_path: str, seq_len: int, split: str = "train"):
        self.seq_len = seq_len

        # 检查文件是否存在
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"找不到数据文件: {bin_path}\n"
                f"请先运行: python scripts/prepare_data.py"
            )

        # 加载内存映射文件（uint16 存储，支持最多 65535 词表大小）
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        n_tokens = len(self.data)
        n_samples = (n_tokens - 1) // seq_len

        print(f"[{split}] 加载数据: {n_tokens:,} tokens → {n_samples:,} 样本 "
              f"(seq_len={seq_len})")

        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ─── 文本数据集（无需预处理，适合小数据）─────────────────────────────────────

class TextDataset(Dataset):
    """
    直接从文本文件加载的数据集（小数据集 <100MB 使用）
    整个数据集加载到内存，支持动态 tokenize
    """

    def __init__(
        self,
        text_files: List[str],
        tokenizer,
        seq_len: int,
        stride: Optional[int] = None,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        stride = stride or seq_len  # 默认无重叠

        print("加载文本并 tokenize...")
        all_ids = [tokenizer.bos_id]  # 以 BOS 开始
        for file_path in text_files:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = tokenizer.encode(line, add_bos=False, add_eos=True)
                    all_ids.extend(ids)

        # 创建滑窗样本
        self.samples: List[torch.Tensor] = []
        total = torch.tensor(all_ids, dtype=torch.long)

        for start in range(0, len(total) - seq_len, stride):
            self.samples.append(total[start:start + seq_len + 1])
            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"✅ 共 {len(all_ids):,} tokens → {len(self.samples):,} 样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.samples[idx]
        return chunk[:-1], chunk[1:]


# ─── 对话数据集 ───────────────────────────────────────────────────────────────

class ChatDataset(Dataset):
    """
    对话格式数据集（用于指令微调 SFT）

    数据格式 (JSON Lines):
    {"instruction": "介绍一下北京", "output": "北京是中国的首都..."}
    {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
    """

    SYSTEM_PROMPT = "你是一个有帮助的中文 AI 助手。"
    USER_TOKEN = "\n<|user|>\n"
    ASST_TOKEN = "\n<|assistant|>\n"
    EOS_TOKEN = "<eos>"

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        seq_len: int = 512,
        ignore_input_loss: bool = True,
    ):
        import json
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.ignore_input_loss = ignore_input_loss
        self.samples: List[Tuple[List[int], List[int]]] = []  # (input_ids, labels)

        for path in data_files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        self._process_item(item)
                    except json.JSONDecodeError:
                        continue

        print(f"✅ 对话数据集: {len(self.samples):,} 条对话")

    def _process_item(self, item: dict):
        if "messages" in item:
            messages = item["messages"]
        elif "instruction" in item:
            messages = [
                {"role": "user",      "content": item["instruction"]},
                {"role": "assistant", "content": item.get("output", item.get("response", ""))},
            ]
        else:
            return

        # 构建对话文本
        text_parts: List[Tuple[str, bool]] = []  # (text, is_response)
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                text_parts.append((content + "\n", False))
            elif role == "user":
                text_parts.append((self.USER_TOKEN + content, False))
            elif role == "assistant":
                text_parts.append((self.ASST_TOKEN + content + self.EOS_TOKEN, True))

        # 编码
        input_ids = [self.tokenizer.bos_id]
        labels = [-100]  # 忽略 BOS 的损失
        is_response_parts = [False]

        for text, is_response in text_parts:
            ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
            input_ids.extend(ids)
            if self.ignore_input_loss and not is_response:
                labels.extend([-100] * len(ids))
            else:
                labels.extend(ids)
            is_response_parts.extend([is_response] * len(ids))

        # 截断到最大长度
        input_ids = input_ids[:self.seq_len]
        labels = labels[:self.seq_len]

        # 构建目标（向右移一位）
        if len(input_ids) < 2:
            return

        x = input_ids[:-1]
        y = labels[1:]  # 预测下一个 token

        if len(x) < 10:
            return

        self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        # Pad 到 seq_len
        pad_len = self.seq_len - 1 - len(x)
        x_tensor = torch.tensor(x + [self.tokenizer.pad_id] * pad_len, dtype=torch.long)
        y_tensor = torch.tensor(y + [-100] * pad_len, dtype=torch.long)
        return x_tensor, y_tensor


# ─── 数据加载器工厂 ───────────────────────────────────────────────────────────

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # 丢弃不完整的最后一批
        persistent_workers=num_workers > 0,
    )


# ─── 数据预处理工具 ───────────────────────────────────────────────────────────

def prepare_binary_dataset(
    input_files: List[str],
    output_dir: str,
    tokenizer,
    train_ratio: float = 0.95,
):
    """
    将文本文件预处理为二进制 token 文件
    每个文档之间插入 EOS token 作为边界，防止生成时串台
    """
    os.makedirs(output_dir, exist_ok=True)
    all_ids = []
    eos_id = tokenizer.eos_id  # 文档分隔符

    for file_path in input_files:
        print(f"处理文件: {file_path}")
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line, add_bos=False, add_eos=True)  # 每行末尾加 EOS
            all_ids.extend(ids)

            if (i + 1) % 100000 == 0:
                print(f"  已处理 {i+1:,} 行, {len(all_ids):,} tokens")

    # 打乱（以句子为单位）
    print("完成 tokenize，准备写入二进制文件...")

    # 拆分训练 / 验证集
    split_point = int(len(all_ids) * train_ratio)
    splits = {
        "train": all_ids[:split_point],
        "val":   all_ids[split_point:],
    }

    for split_name, ids in splits.items():
        arr = np.array(ids, dtype=np.uint16)
        out_path = os.path.join(output_dir, f"{split_name}.bin")
        arr.tofile(out_path)
        print(f"✅ {split_name}: {len(ids):,} tokens → {out_path} "
              f"({os.path.getsize(out_path) / 1024**2:.1f} MB)")

    # 写入元信息
    meta = {
        "total_tokens": len(all_ids),
        "train_tokens": split_point,
        "val_tokens": len(all_ids) - split_point,
        "vocab_size": tokenizer.vocab_size,
    }
    import json
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n📊 数据统计:")
    print(f"  总 token 数: {len(all_ids):,}")
    print(f"  训练集: {split_point:,} tokens")
    print(f"  验证集: {len(all_ids) - split_point:,} tokens")
