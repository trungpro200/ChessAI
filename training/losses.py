import torch
import torch.nn.functional as F

PLANES = 73


def policy_loss(
    policy_logits: torch.Tensor,
    from_sq: torch.Tensor,
    plane: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy on (from_sq, plane) indices. policy_logits: [B, 64, 73]."""
    b = policy_logits.shape[0]
    idx = from_sq * PLANES + plane
    flat = policy_logits.flatten(1, -1)
    return F.cross_entropy(flat, idx)


def policy_accuracy(
    policy_logits: torch.Tensor,
    from_sq: torch.Tensor,
    plane: torch.Tensor,
) -> float:
    b = policy_logits.shape[0]
    flat_idx = policy_logits.flatten(1, -1).argmax(dim=-1)
    target = from_sq * PLANES + plane
    return (flat_idx == target).float().mean().item()


def value_losses(
    value: torch.Tensor,
    z: torch.Tensor,
    z_eval: torch.Tensor,
    has_eval: torch.Tensor,
    w_value: float = 1.0,
    w_eval: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_value_loss, outcome_mse, eval_mse)."""
    value = value.squeeze(-1)
    outcome_mse = F.mse_loss(value, z)
    loss = w_value * outcome_mse

    eval_mse = torch.tensor(0.0, device=value.device)
    if w_eval > 0 and has_eval.any():
        mask = has_eval
        eval_mse = F.mse_loss(value[mask], z_eval[mask])
        loss = loss + w_eval * eval_mse

    return loss, outcome_mse, eval_mse


def compute_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    batch: dict[str, torch.Tensor],
    w_policy: float = 1.0,
    w_value: float = 1.0,
    w_eval: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    p_loss = policy_loss(policy_logits, batch["from_sq"], batch["plane"])
    v_loss, outcome_mse, eval_mse = value_losses(
        value,
        batch["z"],
        batch["z_eval"],
        batch["has_eval"],
        w_value=w_value,
        w_eval=w_eval,
    )
    total = w_policy * p_loss + v_loss
    metrics = {
        "loss": total.item(),
        "policy_loss": p_loss.item(),
        "value_loss": outcome_mse.item(),
        "eval_loss": eval_mse.item() if torch.is_tensor(eval_mse) else float(eval_mse),
        "policy_acc": policy_accuracy(policy_logits, batch["from_sq"], batch["plane"]),
    }
    return total, metrics
