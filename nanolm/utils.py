"""
NanoLM 工具函数
"""

import os
import sys
import json
import random
import time
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def set_seed(seed: int):
    """设置全局随机种子（保证实验可复现）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """获取最佳计算设备"""
    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"使用 GPU: {props.name} ({vram_gb:.1f} GB 显存)")

        # 检查显存是否足够
        if vram_gb < 4:
            print("警告: 显存不足 4GB，训练可能非常缓慢或失败")
        elif vram_gb < 6:
            print("警告: 显存 < 6GB，建议使用 nano 配置并开启梯度检查点")
        else:
            print(f"显存充足，可使用 nano 或 small 配置")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ 使用 Apple Silicon MPS")
    else:
        device = "cpu"
        print("⚠️  未检测到 GPU，使用 CPU 训练（速度非常慢）")

    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """统计模型参数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    """估算模型大小 (MB)，fp32 精度"""
    return count_parameters(model, trainable_only=False) * 4 / 1024**2


def estimate_vram_usage(
    n_params: int,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_embd: int,
    use_amp: bool = True,
    use_grad_checkpoint: bool = True,
) -> Dict[str, float]:
    """
    估算训练时的显存占用 (GB)

    这是一个粗略估算，实际值会有偏差
    """
    bytes_per_param = 2 if use_amp else 4  # fp16 或 fp32

    # 模型参数
    model_mem = n_params * bytes_per_param / 1024**3

    # 梯度（与参数同大小）
    grad_mem = n_params * bytes_per_param / 1024**3

    # 优化器状态（Adam: 2 个动量项，fp32 存储）
    optim_mem = n_params * 8 / 1024**3

    # 激活值（梯度检查点可减少约 sqrt(n_layers) 倍）
    act_per_layer = batch_size * seq_len * n_embd * bytes_per_param
    if use_grad_checkpoint:
        # 只保存检查点层的激活
        act_mem = act_per_layer * 2 / 1024**3  # 大约只需存 2 层
    else:
        act_mem = act_per_layer * n_layers / 1024**3

    total = model_mem + grad_mem + optim_mem + act_mem
    return {
        "model": round(model_mem, 2),
        "gradients": round(grad_mem, 2),
        "optimizer": round(optim_mem, 2),
        "activations": round(act_mem, 2),
        "total_estimated": round(total, 2),
    }


def print_training_config_summary(model, config):
    """打印训练配置摘要"""
    from nanolm.model import ModelConfig
    n_params = count_parameters(model)
    vram = estimate_vram_usage(
        n_params=n_params,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_layers=model.config.n_layers,
        n_embd=model.config.n_embd,
        use_amp=config.use_amp,
        use_grad_checkpoint=config.use_grad_checkpoint,
    )

    print("\n╔══════════════════════════════════════════╗")
    print("║         NanoLM 训练配置摘要              ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║ 模型参数:  {n_params/1e6:>8.2f} M                  ║")
    print(f"║ 模型大小:  {model_size_mb(model):>8.1f} MB (fp32)           ║")
    print(f"║ 序列长度:  {config.seq_len:>8d}                      ║")
    print(f"║ 批大小:    {config.batch_size:>8d} × {config.grad_accum} 累积             ║")
    print(f"║ 有效批大小:{config.effective_batch_size:>8d}                      ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║ 预估显存占用:                            ║")
    print(f"║   模型权重: {vram['model']:>6.2f} GB                     ║")
    print(f"║   梯度:     {vram['gradients']:>6.2f} GB                     ║")
    print(f"║   优化器:   {vram['optimizer']:>6.2f} GB                     ║")
    print(f"║   激活值:   {vram['activations']:>6.2f} GB                     ║")
    print(f"║   总计:     {vram['total_estimated']:>6.2f} GB                     ║")
    print("╚══════════════════════════════════════════╝\n")


class AverageMeter:
    """滑动平均计量器"""
    def __init__(self, window: int = 100):
        self.window = window
        self.values = []

    def update(self, val: float):
        self.values.append(val)
        if len(self.values) > self.window:
            self.values.pop(0)

    @property
    def avg(self) -> float:
        return sum(self.values) / max(1, len(self.values))

    @property
    def latest(self) -> float:
        return self.values[-1] if self.values else 0.0


class Timer:
    """计时器"""
    def __init__(self):
        self.t = time.perf_counter()

    def reset(self):
        self.t = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t


def plot_training_curve(log_path: str, save_path: Optional[str] = None):
    """
    绘制训练曲线（需要 matplotlib）

    Args:
        log_path: training_log.json 文件路径
        save_path: 图片保存路径（None 则显示）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.style as style
    except ImportError:
        print("⚠️  请安装 matplotlib: pip install matplotlib")
        return

    with open(log_path) as f:
        log = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("NanoLM 训练曲线", fontsize=14)

    # 训练损失
    ax = axes[0]
    train_losses = log.get("train_losses", [])
    if train_losses:
        ax.plot(train_losses, alpha=0.6, label="训练损失", color="#3498db")
        # 平滑曲线
        if len(train_losses) > 10:
            smooth = np.convolve(train_losses, np.ones(10)/10, mode="valid")
            ax.plot(range(9, len(train_losses)), smooth, label="平滑", color="#e74c3c")
    ax.set_xlabel("步数 (×log_interval)")
    ax.set_ylabel("损失")
    ax.set_title("训练损失")
    ax.legend()
    ax.grid(alpha=0.3)

    # 验证损失
    ax = axes[1]
    val_losses = log.get("val_losses", [])
    if val_losses:
        ax.plot(val_losses, marker="o", label="验证损失", color="#2ecc71")
        ax.axhline(log.get("best_val_loss", min(val_losses)),
                   linestyle="--", color="#e74c3c", label=f"最佳: {log.get('best_val_loss', 0):.4f}")
    ax.set_xlabel("评估次数")
    ax.set_ylabel("损失")
    ax.set_title("验证损失")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 训练曲线已保存: {save_path}")
    else:
        plt.show()
