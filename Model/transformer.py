import torch
from torch import nn
import torch.nn.functional as F
from .device import device

TARGET_PARAMS = 5_891_914
MAX_PARAMS = TARGET_PARAMS + 300_000

SEQ_LEN = 64
MAX_DIST = 7
NUM_REL = 2 * MAX_DIST + 1
REL_HALF_DIM = 16  # rank + file = head_dim (32)


def get_rel_indices_2d(seq_len: int = SEQ_LEN):
    ranks = torch.arange(seq_len) // 8
    files = torch.arange(seq_len) % 8
    dr = ranks.view(-1, 1) - ranks.view(1, -1)
    df = files.view(-1, 1) - files.view(1, -1)
    dr = torch.clamp(dr, -MAX_DIST, MAX_DIST) + MAX_DIST
    df = torch.clamp(df, -MAX_DIST, MAX_DIST) + MAX_DIST
    return dr, df


class ChessInputEmbedding(nn.Module):
    def __init__(self, input_dim: int = 103, d_model: int = 256):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        half = d_model // 2
        self.rank_embed = nn.Embedding(8, half)
        self.file_embed = nn.Embedding(8, half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != device:
            x = x.to(device)
        if x.dim() < 3:
            x = x.unsqueeze(0)

        x = self.projection(x)
        ranks = torch.arange(SEQ_LEN, device=x.device) // 8
        files = torch.arange(SEQ_LEN, device=x.device) % 8
        pos = torch.cat(
            [self.rank_embed(ranks), self.file_embed(files)], dim=-1
        )
        return x + pos.unsqueeze(0)


class ShawRelativeAttention2D(nn.Module):
    """2D Shaw relative attention with GQA and shared rank/file embeddings."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = n_kv_heads * self.head_dim

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.kv_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.kv_dim, bias=False)

        self.rel_embed_rank = nn.Embedding(NUM_REL, REL_HALF_DIM)
        self.rel_embed_file = nn.Embedding(NUM_REL, REL_HALF_DIM)

        dr, df = get_rel_indices_2d(seq_len)
        self.register_buffer("dr_indices", dr, persistent=False)
        self.register_buffer("df_indices", df, persistent=False)

    def _rel_vectors(self):
        a_rank = self.rel_embed_rank(self.dr_indices)
        a_file = self.rel_embed_file(self.df_indices)
        return a_rank, a_file

    def _expand_kv(self, t: torch.Tensor) -> torch.Tensor:
        B, n_kv, L, d = t.shape
        return t.unsqueeze(2).expand(B, n_kv, self.n_rep, L, d).reshape(
            B, self.n_heads, L, d
        )

    def _rel_logits(self, q, k, a_rank, a_file):
        scale = self.head_dim ** 0.5
        a = torch.cat([a_rank, a_file], dim=-1)

        content_logits = torch.matmul(q, k.transpose(-1, -2))
        rel_logits_k = torch.einsum("bhld,lmd->bhlm", q, a)
        rel_logits_q = torch.einsum("bhmd,lmd->bhlm", k, a)
        rel_pos_pos = torch.einsum("lmd,lmd->lm", a, a)

        return (content_logits + rel_logits_k + rel_logits_q + rel_pos_pos) / scale

    def _rel_out(self, attn, a_rank, a_file):
        a = torch.cat([a_rank, a_file], dim=-1)
        return torch.einsum("bhlm,lmd->bhld", attn, a)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Using -1 forces PyTorch to safely divide d_model by n_heads automatically
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)

        # Instead of 4 einsums, we calculate a POSITION BIAS matrix once
        # rel_indices is (64, 64), we get (64, 64, head_dim)
        a_k = self.rel_embed_k(self.rel_indices) 
        
        # Calculate relative bias: (seq, seq)
        # We simplify Shaw by treating it as a bias added to the attention matrix
        rel_bias = torch.einsum('bhld,lmd->bhlm', q, a_k) 

        # Use the built-in optimized kernel
        # This handles the softmax and scaling internally and MUCH faster
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=rel_bias, # Add the relative info here
            dropout_p=0.0 if not self.training else 0.1
        )
        
        return out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)


class SwiGLUFFN(nn.Module):
    """Param-matched SwiGLU: 2*d*h + h*d + biases ≈ standard 256→1024→256 FFN."""

    def __init__(self, d_model: int = 256, hidden_mult: float = 2.66):
        super().__init__()
        hidden = int(d_model * hidden_mult)
        self.w_gate = nn.Linear(d_model, hidden)
        self.w_up = nn.Linear(d_model, hidden)
        self.w_down = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.w_down.weight)
        nn.init.zeros_(self.w_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ChessTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        ffn_hidden_mult: float = 2.66,
    ):
        super().__init__()
        self.attn = ShawRelativeAttention2D(d_model, n_heads, n_kv_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_hidden_mult)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ChessHeads(nn.Module):
    POLICY_BOTTLENECK = 57  # 256*k + 73*k = 256*73

    def __init__(self, d_model: int = 256, value_hidden: int = 160):
        super().__init__()
        k = self.POLICY_BOTTLENECK
        self.policy_u = nn.Linear(d_model, k, bias=False)
        self.policy_v = nn.Linear(k, 73, bias=True)

        self.value_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.value_net = nn.Sequential(
            nn.Linear(d_model, value_hidden),
            nn.Mish(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, body_out: torch.Tensor):
        policy_logits = self.policy_v(self.policy_u(body_out))

        scores = torch.einsum("bld,d->bl", body_out, self.value_query)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (body_out * weights).sum(dim=1)
        value_logits = self.value_net(pooled)

        return policy_logits, value_logits


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
