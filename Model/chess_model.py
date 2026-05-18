from .transformer import ChessHeads, ChessInputEmbedding, ChessTransformerBlock, TARGET_PARAMS, MAX_PARAMS, count_parameters
import torch
from torch import nn
from .device import device


class ChessModel(nn.Module):
    """Transformer chess net for [B, 64, 103] → policy [B, 64, 73], value [B, 1].

    Defaults: d_model=256, 9 layers, 8 query heads, 2 KV heads (GQA).
    Parameter budget: TARGET_PARAMS (legacy 8-layer model), up to MAX_PARAMS (+300k).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 9,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        token_dim: int = 103,
        value_hidden: int = 224,
        ffn_hidden_mult: float = 2.85,
    ) -> None:
        super().__init__()

        self.CIE = ChessInputEmbedding(input_dim=token_dim, d_model=d_model)
        self.blocks = nn.Sequential(
            *[
                ChessTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    ffn_hidden_mult=ffn_hidden_mult,
                )
                for _ in range(n_layers)
            ]
        )
        self.heads = ChessHeads(d_model, value_hidden=value_hidden)

        n_params = count_parameters(self)
        if n_params > MAX_PARAMS:
            raise ValueError(
                f"ChessModel has {n_params:,} parameters, exceeds MAX_PARAMS {MAX_PARAMS:,}"
            )

        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.CIE(x)
        x = self.blocks(x)
        return self.heads(x)


def verify_model(model: ChessModel | None = None) -> dict:
    """Forward + parameter budget checks."""
    if model is None:
        model = ChessModel()
    model.eval()
    x = torch.randn(4, 64, 103, device=next(model.parameters()).device)
    with torch.no_grad():
        policy, value = model(x)
    n = count_parameters(model)
    return {
        "params": n,
        "target": TARGET_PARAMS,
        "max": MAX_PARAMS,
        "policy_shape": tuple(policy.shape),
        "value_shape": tuple(value.shape),
        "within_budget": n <= MAX_PARAMS,
    }
