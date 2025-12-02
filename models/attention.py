import torch
import torch.nn as nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = config['head_dim']
        self.att_dim = self.n_heads * self.head_dim

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.att_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.att_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.att_dim, bias=True)

        self.output_proj = nn.Linear(self.att_dim, self.hidden_size)

        # Initialize as before
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, x, k=None, v=None):
        # Default to self-attention if k/v not provided
        k = x if k is None else k
        v = x if v is None else v

        B, N, _ = x.shape  # x shape: (B, N, hidden_size)

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, N, att_dim)
        k = self.k_proj(k)  # (B, N, att_dim)
        v = self.v_proj(v)  # (B, N, att_dim)

        # Split into multi-head attention
        q = rearrange(q, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)

        # Attention scores
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = torch.softmax(att, dim=-1)

        # Weighted sum
        out = torch.matmul(att, v)  # (B, n_heads, N, head_dim)
        out = rearrange(out, 'b n_h n h_dim -> b n (n_h h_dim)')  # (B, N, att_dim)

        return self.output_proj(out)  # (B, N, hidden_size)