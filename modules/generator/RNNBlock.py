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

class GroupedRNN(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, n_groups: int, rnn_type: str = 'gru', bidirectional: bool = False):
        super().__init__()
        assert n_embd % n_groups == 0, "n_embd must be divisible by n_groups"
        
        self.n_groups = n_groups
        self.group_size = n_embd // n_groups
        
        rnn_type = rnn_type.lower()
        if rnn_type not in ['gru', 'lstm']:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        
        rnn_class = nn.GRU if rnn_type == 'gru' else nn.LSTM
        
        self.rnns = nn.ModuleList()
        for _ in range(self.n_groups):
            if bidirectional:
                rnn_hidden_size = self.group_size // 2
            else:
                rnn_hidden_size = self.group_size

            self.rnns.append(
                rnn_class(
                    input_size=self.group_size,
                    hidden_size=rnn_hidden_size,
                    num_layers=n_layer,
                    bias=False,
                    batch_first=True,
                    bidirectional=bidirectional
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, n_embd)
        B, T, D = x.shape
        
        # (B, T, D) -> (B, T, G, group_size)
        x_groups = x.view(B, T, self.n_groups, self.group_size)
        
        rnn_outputs = []
        for g in range(self.n_groups):
            x_g = x_groups[:, :, g, :]
            rnn_out_g, _ = self.rnns[g](x_g)
            rnn_outputs.append(rnn_out_g)
            
        rnn_output = torch.cat(rnn_outputs, dim=-1)
        return rnn_output

class RNNBlock(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, n_groups: int, rnn_type: str = 'gru', bidirectional: bool = False):
        super().__init__()
        self.rnn_norm = RMSNorm(n_embd)
        
        self.rnn = GroupedRNN(n_embd, n_layer, n_groups, rnn_type, bidirectional)
        
        self.ffn_norm = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.rnn(self.rnn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x

if __name__ == '__main__':
    n_embd = 512
    batch_size = 4
    seq_len = 100
    n_groups = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_tensor = torch.randn(batch_size, seq_len, n_embd).to(device)

    n_layer_rnn = 2  

    model_gru = RNNBlock(
        n_embd=n_embd,
        n_layer=n_layer_rnn,
        n_groups=n_groups,
        rnn_type='gru'
    )
    model_gru.to(device)
    model_gru.eval()
    with torch.no_grad():
        output_gru = model_gru(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape (GRU): {output_gru.shape}")
    assert output_gru.shape == (batch_size, seq_len, n_embd)

    model_bigru = RNNBlock(
        n_embd=n_embd,
        n_layer=n_layer_rnn,
        n_groups=n_groups,
        rnn_type='gru',
        bidirectional=True
    )
    model_bigru.to(device)
    model_bigru.eval()
    with torch.no_grad():
        output_bigru = model_bigru(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape (BiGRU): {output_bigru.shape}")
    assert output_bigru.shape == (batch_size, seq_len, n_embd)