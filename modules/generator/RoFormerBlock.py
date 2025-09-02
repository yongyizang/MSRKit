import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        hidden_dim = 4 * n_embd
        n_hidden = int(2 * hidden_dim / 3)
        self.c_fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.silu(self.c_fc1(x)) * self.c_fc2(x))

class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, rope_base: int = 10000):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        rope_cache = self._build_rope_cache(max_seq_len, self.head_dim, rope_base)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        q = self._apply_rope(q, self.rope_cache)
        k = self._apply_rope(k, self.rope_cache)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(y)

    @staticmethod
    def _build_rope_cache(seq_len: int, head_dim: int, base: int = 10000) -> torch.Tensor:
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        seq_idx = torch.arange(seq_len, dtype=torch.float32)
        idx_theta = torch.outer(seq_idx, theta)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        return cache

    @staticmethod
    def _apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        rope_cache = rope_cache[:T]
        x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(1, T, 1, x_shaped.size(3), 2)
        x_out = torch.stack(
            [
                x_shaped[..., 0] * rope_cache[..., 0] - x_shaped[..., 1] * rope_cache[..., 1],
                x_shaped[..., 1] * rope_cache[..., 0] + x_shaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        return x_out.flatten(3).type_as(x)

class RoFormerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, rope_base: int = 10000):
        super().__init__()
        self.att_norm = RMSNorm(n_embd)
        self.att = SelfAttention(n_embd, n_head, max_seq_len, rope_base)
        self.ffn_norm = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x

if __name__ == '__main__':
    n_head = 8
    n_embd = 512
    max_seq_len = 256
    
    model = RoFormerBlock(
        n_embd=n_embd,
        n_head=n_head,
        max_seq_len=max_seq_len,
        rope_base=10000
    )

    batch_size = 4
    seq_len = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_tensor = torch.randn(batch_size, seq_len, n_embd).to(device)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, n_embd)