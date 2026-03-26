from .transformer import *
import torch

class ChessModel(nn.Module):
    """ChessModel

    Args:
        tensor: return logits(policy, value)
    """
    def __init__(self, d_model=256, n_layers=8, n_heads=8, token_dim = 55) -> None:
        super().__init__()
        
        self.CIE = ChessInputEmbedding(input_dim=token_dim, d_model=d_model)
        self.blocks = nn.Sequential(*[ChessTransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.heads = ChessHeads(d_model) # -> logits(policy, value)
        
        self.to(device)
    
    def forward(self, x):
        # Value-up (increase dim)
        x = self.CIE(x)
        x = self.blocks(x)
        
        policy_logits, value_logits = self.heads(x)
        
        return policy_logits, value_logits
        
        
        
        