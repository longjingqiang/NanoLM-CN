"""
NanoLM 模型架构 - GPT 风格的 Decoder-Only Transformer
专为个人 GPU (6GB 显存) 优化设计
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 8000          # 词表大小
    max_seq_len: int = 512          # 最大序列长度
    n_layers: int = 6               # Transformer 层数
    n_heads: int = 8                # 注意力头数
    n_embd: int = 256               # 嵌入维度
    ffn_ratio: float = 4.0          # FFN 扩展比例
    dropout: float = 0.1            # Dropout 率
    use_bias: bool = False          # 是否使用偏置
    rope_base: int = 10000          # RoPE 基础频率
    tie_embeddings: bool = True     # 是否共享输入输出嵌入
    norm_eps: float = 1e-5          # LayerNorm epsilon

    @property
    def n_ffn(self) -> int:
        """FFN 中间层大小 (对齐到 256 的倍数)"""
        raw = int(self.n_embd * self.ffn_ratio)
        return (raw + 255) // 256 * 256

    @property
    def n_params(self) -> int:
        """估算参数量"""
        embed = self.vocab_size * self.n_embd
        attn = self.n_layers * 4 * self.n_embd * self.n_embd
        ffn = self.n_layers * 2 * self.n_embd * self.n_ffn
        total = embed + attn + ffn
        if not self.tie_embeddings:
            total += self.vocab_size * self.n_embd
        return total

    def __str__(self):
        return (f"ModelConfig(layers={self.n_layers}, heads={self.n_heads}, "
                f"embd={self.n_embd}, vocab={self.vocab_size}, "
                f"params≈{self.n_params/1e6:.1f}M)")


# ─── 预设配置 ─────────────────────────────────────────────────────────────────

PRESET_CONFIGS = {
    "nano": ModelConfig(
        vocab_size=8000, max_seq_len=512, n_layers=6,
        n_heads=8, n_embd=256, dropout=0.1
    ),  # ~10M params，最适合 6GB 显存快速实验

    "small": ModelConfig(
        vocab_size=8000, max_seq_len=512, n_layers=8,
        n_heads=8, n_embd=512, dropout=0.1
    ),  # ~35M params，6GB 显存可训练

    "medium": ModelConfig(
        vocab_size=8000, max_seq_len=512, n_layers=12,
        n_heads=12, n_embd=768, dropout=0.1
    ),  # ~117M params，需要梯度检查点

    "large": ModelConfig(
        vocab_size=16000, max_seq_len=1024, n_layers=16,
        n_heads=16, n_embd=1024, dropout=0.1
    ),  # ~350M params，需要高端 GPU
}


# ─── 旋转位置编码 (RoPE) ───────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """旋转位置编码 - 比绝对位置编码外推性更好"""

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.base = base
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ─── RMS Norm ─────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """RMS 归一化 - 比 LayerNorm 更高效"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ─── 多头因果自注意力 ──────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """多头因果自注意力，带 RoPE 位置编码"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout

        # QKV 投影合并为一个矩阵以提升效率
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.use_bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.rotary = RotaryEmbedding(self.head_dim, config.rope_base, config.max_seq_len)

        # 因果掩码（下三角）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 计算 Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 重塑为多头格式 (B, heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置编码
        cos, sin = self.rotary(q, T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # 尝试使用 Flash Attention (PyTorch 2.0+)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 手动实现因果注意力
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_out = attn @ v

        # 合并多头输出
        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# ─── 前馈网络 (SwiGLU) ────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """SwiGLU 前馈网络 - 性能优于标准 FFN"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.n_ffn, bias=config.use_bias)
        self.up_proj   = nn.Linear(config.n_embd, config.n_ffn, bias=config.use_bias)
        self.down_proj = nn.Linear(config.n_ffn, config.n_embd, bias=config.use_bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * silu(gate) ⊙ up
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """单个 Transformer 块 (Pre-Norm 架构)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn  = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd, config.norm_eps)
        self.ffn   = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # 残差 + 注意力
        x = x + self.ffn(self.norm2(x))    # 残差 + FFN
        return x


# ─── 主模型 ───────────────────────────────────────────────────────────────────

class NanoLM(nn.Module):
    """
    NanoLM - 轻量级中文语言模型

    架构: GPT 风格 Decoder-Only Transformer
    - RoPE 位置编码（代替绝对位置编码）
    - RMSNorm（代替 LayerNorm）
    - SwiGLU FFN（代替 ReLU FFN）
    - Flash Attention (PyTorch 2.0+)
    - 可选梯度检查点（节省显存）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.embd_drop  = nn.Dropout(config.dropout)
        self.blocks     = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_final = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head    = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定：共享输入和输出嵌入
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # 参数初始化
        self.apply(self._init_weights)
        # 对残差投影进行缩放初始化（GPT-2 方式）
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        print(f"NanoLM 初始化完成: {config}")
        print(f"参数量: {self.num_params / 1e6:.2f}M")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"序列长度 {T} 超过最大限制 {self.config.max_seq_len}"
        )

        # Token 嵌入
        x = self.embd_drop(self.embedding(input_ids))

        # 通过各 Transformer 块
        for block in self.blocks:
            x = block(x)

        x = self.norm_final(x)

        if targets is not None:
            # 训练模式：计算交叉熵损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        else:
            # 推理模式：只计算最后一个位置
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_at_eos: bool = True,       # 遇到 EOS 立即停止（修复串台）
        eos_token_id: int = 3,          # SPECIAL_TOKENS["<eos>"] = 3
        min_new_tokens: int = 0,        # 最少生成 token 数（在此数量之前禁止 EOS）
    ) -> torch.Tensor:
        """
        自回归文本生成

        Args:
            input_ids: 输入 token ids, shape (B, T)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数（越低越保守）
            top_k: Top-K 采样
            top_p: Top-P (核采样)
            repetition_penalty: 重复惩罚（>1 减少重复）
            stop_at_eos: 遇到 EOS token 时停止（防止串台）
            eos_token_id: EOS token 的 id
            min_new_tokens: 最少生成 token 数（在此数量之前禁止 EOS）
        """
        self.eval()
        # 记录每个样本是否已生成完毕
        B = input_ids.shape[0]
        finished = [False] * B

        tokens_generated = 0
        for _ in range(max_new_tokens):
            # 如果所有样本都已完成，提前退出
            if all(finished):
                break

            # 如果超过最大长度，截断
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                       else input_ids[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # 重复惩罚
            if repetition_penalty != 1.0:
                for b in range(B):
                    for token_id in set(input_ids[b].tolist()):
                        logits[b, token_id] /= repetition_penalty

            # 温度缩放
            logits = logits / max(temperature, 1e-8)

            # 在前 min_new_tokens 个 token 中禁止 EOS
            if tokens_generated < min_new_tokens:
                logits[:, eos_token_id] = float("-inf")

            # Top-K 过滤
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, [-1]]] = float("-inf")

            # Top-P (核采样) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_remove] = float("-inf")
                logits.scatter_(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 已完成的样本固定输出 PAD
            for b in range(B):
                if finished[b]:
                    next_token[b, 0] = 0  # pad_id

            input_ids = torch.cat([input_ids, next_token], dim=1)
            tokens_generated += 1

            # 检测 EOS
            if stop_at_eos:
                for b in range(B):
                    if next_token[b, 0].item() == eos_token_id:
                        finished[b] = True

        return input_ids

    def enable_gradient_checkpointing(self):
        """启用梯度检查点以节省显存（以重算代替存储）"""
        from torch.utils.checkpoint import checkpoint

        for block in self.blocks:
            original_forward = block.forward

            def make_checkpointed(orig_fwd):
                def checkpointed_forward(x):
                    return checkpoint(orig_fwd, x, use_reentrant=False)
                return checkpointed_forward

            block.forward = make_checkpointed(original_forward)
        print("✅ 梯度检查点已启用（节省约 40% 显存，训练速度略降）")

    @classmethod
    def from_pretrained(cls, path: str) -> "NanoLM":
        """从检查点加载模型"""
        import json, os
        config_path = os.path.join(path, "config.json")
        with open(config_path) as f:
            cfg_dict = json.load(f)
        config = ModelConfig(**cfg_dict)
        model = cls(config)
        ckpt_path = os.path.join(path, "model.pt")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str):
        """保存模型到目录"""
        import json, os, dataclasses
        os.makedirs(path, exist_ok=True)
        cfg = dataclasses.asdict(self.config)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        print(f"✅ 模型已保存至 {path}")
