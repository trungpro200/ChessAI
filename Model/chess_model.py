import torch
from torch import nn
from .device import device
from .transformer import TARGET_PARAMS, MAX_PARAMS, count_parameters


def gn(channels: int) -> nn.GroupNorm:
    # Stable even with smaller batch sizes
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ChessCNNBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = gn(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = gn(channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class ChessCNNHeads(nn.Module):
    """
    Input:  [B, C, 8, 8]
    Output: policy [B, 64, 73], value [B, 1]
    """
    def __init__(self, d_model: int, value_hidden: int = 128) -> None:
        super().__init__()

        # Policy: one 73-logit vector per square
        self.policy = nn.Sequential(
            nn.Conv2d(d_model, 73, kernel_size=1, bias=True),
        )

        # Value: board-wide scalar
        self.value = nn.Sequential(
            nn.Conv2d(d_model, value_hidden, kernel_size=1, bias=False),
            gn(value_hidden),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.value_fc = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor):
        policy = self.policy(x)                 # [B, 73, 8, 8]
        policy = policy.permute(0, 2, 3, 1)     # [B, 8, 8, 73]
        policy = policy.reshape(x.size(0), 64, 73)

        v = self.value(x).flatten(1)            # [B, value_hidden]
        value = torch.tanh(self.value_fc(v))    # [B, 1]
        return policy, value


class ChessModel(nn.Module):
    """
    CNN chess net for [B, 64, 103] -> policy [B, 64, 73], value [B, 1].

    Good starting point for RTX 2060 12GB:
      d_model=192, n_blocks=8
    Smaller / faster:
      d_model=128, n_blocks=6
    """
    def __init__(
        self,
        d_model: int = 192,
        n_blocks: int = 8,
        token_dim: int = 103,
        value_hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_dim = token_dim

        self.stem = nn.Sequential(
            nn.Conv2d(token_dim, d_model, kernel_size=3, padding=1, bias=False),
            gn(d_model),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[ChessCNNBlock(d_model, dropout=dropout) for _ in range(n_blocks)]
        )

        self.heads = ChessCNNHeads(d_model, value_hidden=value_hidden)

        n_params = count_parameters(self)
        if n_params > MAX_PARAMS:
            raise ValueError(
                f"ChessModel has {n_params:,} parameters, exceeds MAX_PARAMS {MAX_PARAMS:,}"
            )

        self.to(device)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 64, 103]
        """
        if x.ndim != 3 or x.shape[1] != 64 or x.shape[2] != self.token_dim:
            raise ValueError(f"Expected input [B, 64, {self.token_dim}], got {tuple(x.shape)}")

        # [B, 64, 103] -> [B, 103, 8, 8]
        x = x.view(x.size(0), 8, 8, self.token_dim).permute(0, 3, 1, 2).contiguous()

        x = self.stem(x)
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