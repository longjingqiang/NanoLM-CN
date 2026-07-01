"""
NanoLM 训练引擎
专为个人 GPU (6GB 显存) 深度优化：
- 混合精度训练 (fp16/bf16)
- 梯度累积（模拟大批次）
- 梯度检查点（显存换速度）
- 动态学习率调度（Cosine with Warmup）
- 自动保存检查点和恢复
"""

import os
import time
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast   # 新 API，消除 FutureWarning
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path


# ─── 训练配置 ─────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # 数据路径
    data_dir: str = "data/processed"
    output_dir: str = "checkpoints"
    tokenizer_path: str = "checkpoints/tokenizer"

    # 模型
    model_preset: str = "nano"    # nano / small / medium / large

    # 训练超参数
    max_iters: int = 50000        # 总训练步数
    batch_size: int = 16          # 每步实际 batch size
    grad_accum: int = 4           # 梯度累积步数（有效 batch = batch_size × grad_accum）
    seq_len: int = 512            # 序列长度

    # 学习率
    lr: float = 3e-4              # 峰值学习率
    lr_min: float = 3e-5          # 最低学习率
    warmup_iters: int = 1000      # 预热步数
    lr_decay_iters: int = 45000   # 学习率衰减至终点的步数

    # 优化器
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # 早停
    early_stop_patience: int = 5            # 验证损失连续多少次不改善则早停
    early_stop_min_delta: float = 0.0001     # 验证损失改善的最小阈值

    # 显存优化
    use_amp: bool = True          # 自动混合精度
    amp_dtype: str = "bf16"       # "fp16" 或 "bf16"（Ampere+ GPU 用 bf16）
    use_grad_checkpoint: bool = True  # 梯度检查点（节省 ~40% 显存）
    compile_model: bool = False   # torch.compile（PyTorch 2.0+，更快但编译慢）

    # 日志与保存
    log_interval: int = 50        # 每 N 步打印日志
    eval_interval: int = 500      # 每 N 步评估
    save_interval: int = 2000     # 每 N 步保存检查点
    eval_iters: int = 100         # 评估时跑多少步
    keep_checkpoints: int = 3     # 保留最近 N 个检查点

    # 其他
    seed: int = 42
    num_workers: int = 2          # DataLoader 工作进程数（Windows 建议设 0）
    pin_memory: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


# ─── 学习率调度 ───────────────────────────────────────────────────────────────

def get_lr(step: int, config: TrainConfig) -> float:
    """Cosine 学习率调度 with 线性预热"""
    # 预热阶段：线性增大
    if step < config.warmup_iters:
        return config.lr * step / max(1, config.warmup_iters)

    # 衰减结束后：保持最低学习率
    if step > config.lr_decay_iters:
        return config.lr_min

    # Cosine 衰减
    progress = (step - config.warmup_iters) / max(1, config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.lr_min + coeff * (config.lr - config.lr_min)


# ─── 显存监控 ─────────────────────────────────────────────────────────────────

def get_gpu_memory_info() -> Dict[str, float]:
    """获取 GPU 显存使用情况 (GB)"""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "reserved":  torch.cuda.memory_reserved()  / 1024**3,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
    }


def print_gpu_status():
    """打印 GPU 状态"""
    if not torch.cuda.is_available():
        print("⚠️  未检测到 CUDA GPU")
        return
    props = torch.cuda.get_device_properties(0)
    mem = get_gpu_memory_info()
    total_gb = props.total_memory / 1024**3
    print(f"🖥️  GPU: {props.name} | "
          f"显存: {mem.get('allocated', 0):.1f}/"
          f"{total_gb:.1f} GB | "
          f"峰值: {mem.get('max_allocated', 0):.1f} GB")


# ─── 训练器 ───────────────────────────────────────────────────────────────────

class Trainer:
    """NanoLM 训练器"""

    def __init__(self, model, train_loader, val_loader, config: TrainConfig, device: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # 梯度检查点
        if config.use_grad_checkpoint:
            model.enable_gradient_checkpointing()

        # 编译模型（PyTorch 2.0+）
        if config.compile_model:
            print("⚙️  编译模型中 (torch.compile)... 首次训练会慢，请耐心等待")
            self.model = torch.compile(model)

        # 优化器（分组应用权重衰减）
        self.optimizer = self._build_optimizer()

        # 混合精度
        self.use_amp = config.use_amp and torch.cuda.is_available()
        if self.use_amp:
            amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
            # bf16 不需要 GradScaler（bf16 不会下溢）
            self.scaler = GradScaler("cuda") if config.amp_dtype == "fp16" else None
            self.amp_dtype = amp_dtype
            self.amp_device = "cuda"
            print(f"✅ 混合精度训练: {config.amp_dtype}")
        else:
            self.scaler = None
            self.amp_dtype = torch.float32
            self.amp_device = "cpu" if not torch.cuda.is_available() else "cuda"

        # 训练状态
        self.step = 0
        self.best_val_loss = float("inf")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self._early_stop_counter = 0  # 早停计数器

        # 检查点保存队列
        self._saved_checkpoints: List[str] = []

        os.makedirs(config.output_dir, exist_ok=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建 AdamW 优化器（对不需要权重衰减的参数分组）"""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # 偏置和归一化层不应用权重衰减
            if param.ndim < 2 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        cfg = self.config
        groups = [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=True
            if torch.cuda.is_available() else False
        )
        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"优化器分组: 权重衰减={n_decay/1e6:.2f}M 参数, "
              f"无衰减={n_no_decay/1e6:.2f}M 参数")
        return optimizer

    def _update_lr(self):
        """更新学习率"""
        lr = get_lr(self.step, self.config)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """在验证集上评估模型"""
        self.model.eval()
        losses = []
        val_iter = iter(self.val_loader)

        for _ in range(self.config.eval_iters):
            try:
                x, y = next(val_iter)
            except StopIteration:
                val_iter = iter(self.val_loader)
                x, y = next(val_iter)

            x, y = x.to(self.device), y.to(self.device)
            with autocast(self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(x, y)
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        self.model.train()
        return avg_loss

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """单个训练步骤（支持梯度累积）"""
        x, y = x.to(self.device), y.to(self.device)

        with autocast(self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
            _, loss = self.model(x, y)
            loss = loss / self.config.grad_accum  # 梯度累积归一化

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.grad_accum  # 返回未缩放的损失

    def optimizer_step(self):
        """执行优化器步骤（梯度裁剪 + 更新）"""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(self, name: str, extra: dict = None):
        """保存检查点"""
        path = os.path.join(self.config.output_dir, f"{name}.pt")
        state = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }
        if self.scaler:
            state["scaler_state"] = self.scaler.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

        # 管理检查点数量
        self._saved_checkpoints.append(path)
        if len(self._saved_checkpoints) > self.config.keep_checkpoints:
            old = self._saved_checkpoints.pop(0)
            if os.path.exists(old) and "best" not in old:
                os.remove(old)

        return path

    def load_checkpoint(self, path: str):
        """恢复检查点"""
        print(f"加载检查点: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if self.scaler and "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        print(f"✅ 已恢复到步骤 {self.step}, 最佳验证损失: {self.best_val_loss:.4f}")

    def train(self):
        """主训练循环"""
        cfg = self.config
        print("\n" + "=" * 60)
        print("🚀 开始训练 NanoLM")
        print(f"   设备: {self.device}")
        print(f"   模型: {cfg.model_preset}")
        print(f"   批大小: {cfg.batch_size} × 累积 {cfg.grad_accum} = {cfg.effective_batch_size}")
        print(f"   序列长度: {cfg.seq_len}")
        print(f"   总步数: {cfg.max_iters:,}")
        print(f"   学习率: {cfg.lr} → {cfg.lr_min}")
        print("=" * 60 + "\n")
        print_gpu_status()

        self.model.train()
        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        accum_count = 0
        t0 = time.time()

        for step in range(self.step, cfg.max_iters):
            self.step = step

            # 更新学习率
            lr = self._update_lr()

            # ── 梯度累积 ──
            for micro_step in range(cfg.grad_accum):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                loss = self.train_step(x, y)
                accum_loss += loss
                accum_count += 1

            # ── 优化器更新 ──
            self.optimizer_step()

            # ── 日志 ──
            if step % cfg.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                avg_loss = accum_loss / accum_count
                self.train_losses.append(avg_loss)
                accum_loss = accum_count = 0

                tokens_per_sec = (
                    cfg.log_interval * cfg.effective_batch_size * cfg.seq_len / dt
                )
                mem = get_gpu_memory_info()
                print(
                    f"步骤 {step:6d}/{cfg.max_iters} | "
                    f"损失: {avg_loss:.4f} | "
                    f"lr: {lr:.2e} | "
                    f"{tokens_per_sec/1000:.1f}K tok/s | "
                    f"显存: {mem.get('allocated', 0):.1f}GB"
                )

            # ── 评估 ──
            if step > 0 and step % cfg.eval_interval == 0:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                print(f"  📊 验证损失: {val_loss:.4f} (最佳: {self.best_val_loss:.4f})")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._early_stop_counter = 0  # 重置早停计数器
                    self.save_checkpoint("best")
                    print(f"  ✅ 新的最佳模型已保存！")
                else:
                    self._early_stop_counter += 1
                    if self._early_stop_counter >= self.config.early_stop_patience:
                        print(f"  ⏳ 达到早停条件，连续 {self.config.early_stop_patience} 次验证损失未改善")
                        break

            # ── 定期保存 ──
            if step > 0 and step % cfg.save_interval == 0:
                path = self.save_checkpoint(f"step_{step:07d}")
                print(f"  💾 检查点已保存: {path}")

        # 训练结束
        print("\n" + "=" * 60)
        print("🎉 训练完成！")
        print(f"   最佳验证损失: {self.best_val_loss:.4f}")
        final_path = self.save_checkpoint("final")
        print(f"   最终模型: {final_path}")
        print_gpu_status()
        print("=" * 60)

        # 保存训练曲线
        self._save_training_log()

    def _save_training_log(self):
        """保存训练日志"""
        log = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_steps": self.step,
        }
        path = os.path.join(self.config.output_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"📈 训练日志已保存: {path}")
