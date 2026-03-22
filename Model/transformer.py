import torch
from torch import nn
import torch.nn.functional as F 
from .device import device

class ChessInputEmbedding(nn.Module):
    def __init__(self, input_dim=112, d_model=256):
        super().__init__()
        # Lớp Linear để biến đổi 112 chiều thô thành d_model chiều
        self.projection = nn.Linear(input_dim, d_model)
        
        self.pos_offset = nn.Parameter(torch.randn(64, d_model))

    def forward(self, x:torch.Tensor):
        # x shape: [batch_size, 64, 112]
        if x.device != device:
            x = x.to(device)
        
        if len(x.shape)<3:
            x = x.unsqueeze(0)
        
        x = self.projection(x) # Chuyển thành [batch_size, 64, d_model]
        
        # Cộng thêm thông tin vị trí tuyệt đối
        x = x + self.pos_offset
        return x

class ShawRelativeAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, max_dist=7, seq_len = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_dist = max_dist
        
        # Ma trận chiếu cho nội dung (Standard QKV)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Bảng nhúng vị trí tương đối (Shaw et al. 2018)
        num_rel_pos = 2 * max_dist + 1
        
        self.rel_embed_q = nn.Embedding(num_rel_pos, self.head_dim)
        self.rel_embed_k = nn.Embedding(num_rel_pos, self.head_dim)
        self.rel_embed_v = nn.Embedding(num_rel_pos, self.head_dim)
        
        self.rel_indices = self.get_rel_indices(seq_len).to(device)
        
        self.to(device)

    def get_rel_indices(self, seq_len):
        """Tạo ma trận chỉ số từ 0 đến 2k [4, 7]"""
        # Tạo dải chỉ số [0, 1, ..., 63]
        range_vec = torch.arange(seq_len)
        # Tính khoảng cách tương đối giữa mọi cặp (i, j)
        rel_indices = range_vec.view(-1, 1) - range_vec.view(1, -1)
        # Cắt (clip) vào khoảng [-k, k] và dịch chuyển sang số dương [0, 2k] [6, 8]
        rel_indices = torch.clamp(rel_indices, -self.max_dist, self.max_dist)
        return rel_indices + self.max_dist

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Chiếu nội dung và tách đầu (Heads)
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Lấy vector nhúng từ nn.Embedding
        # rel_indices shape: (64, 64)
        # a_k, a_v, a_q shape: (64, 64, 32)

        a_q = self.rel_embed_q(self.rel_indices)
        a_k = self.rel_embed_k(self.rel_indices) 
        a_v = self.rel_embed_v(self.rel_indices)

         # 3. Tính toán Attention Logits (Mở rộng đầy đủ)
        # Tích nội dung: q * k^T
        content_logits = torch.matmul(q, k.transpose(-1, -2)) # [B, H, 64, 64]
        
        # Tương tác Query-Position: q * a_k^T
        rel_logits_k = torch.einsum('bhld,lmd->bhlm', q, a_k)
        
        # Tương tác Position-Key: a_q * k^T
        rel_logits_q = torch.einsum('bhmd,lmd->bhlm', k, a_q)
        
        # Tương tác Position-Position: a_q * a_k^T
        rel_pos_pos = torch.einsum('lmd,lmd->lm', a_q, a_k) 
        
        # Tổng hợp Logits:
        logits = (content_logits + rel_logits_k + rel_logits_q + rel_pos_pos) / (self.head_dim ** 0.5) 
        attn = F.softmax(logits, dim=-1) # alpha

        # 4. Tính toán Output
        # Tổng nội dung: attn * v
        content_out = torch.matmul(attn, v)
        # Tổng vị trí tương đối: attn * a_v
        rel_out = torch.einsum('bhlm,lmd->bhld', attn, a_v)
        
        out = content_out + rel_out
        
        return out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
    
class ChessTransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.attn = ShawRelativeAttention(d_model, n_heads) 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.Mish(), # Mish out-perform ReLU
            nn.Linear(d_model * 4, d_model)
        )
        
        self.to(device)

    def forward(self, x):
        x = x + self.attn(x)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

class ChessHeads(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # Policy Head: Chiếu body output thành Query (ô đi) và Key (ô đến)
        self.policy_head = nn.Linear(d_model, 73) # Shape = [64,73]
        
        # Value Head: Dự đoán value từ góc nhìn của người chơi bên trắng
        self.value_net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Mish(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self.to(device)

    def forward(self, body_out):
        # body_out: [Batch, 64, d_model]
        
        # 1. Tính Policy Logits (64x73)
        policy_logits = self.policy_head(body_out) # [B, 64, 73]
        
        # 2. Tính Value Logits
        pooled = body_out.mean(dim=1) # Mean pooling [B, D]
        value_logits = self.value_net(pooled) # [B, 3]
        
        return policy_logits, value_logits