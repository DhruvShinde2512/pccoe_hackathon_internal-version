"""Small GPU autoencoder and exact XGBoost TreeSHAP helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from xgboost import DMatrix, XGBClassifier


class TabularAutoencoder(nn.Module):
    """Compact reconstruction network for standardized tabular features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of standardized transaction features."""
        return self.network(values)


def fit_autoencoder(
    normal_features: np.ndarray,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    device: str | None = None,
    seed: int = 42,
) -> tuple[dict[str, Any], str]:
    """Fit only on the known routine baseline and return portable CPU weights."""
    if normal_features.ndim != 2 or len(normal_features) < 8:
        raise ValueError("autoencoder requires at least eight two-dimensional normal rows")
    if not np.isfinite(normal_features).all():
        raise ValueError("normal features contain non-finite values")
    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = TabularAutoencoder(normal_features.shape[1]).to(actual_device)
    tensor = torch.as_tensor(normal_features, dtype=torch.float32, device=actual_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tensor), device=actual_device)
        for indices in order.split(batch_size):
            batch = tensor[indices]
            loss = nn.functional.mse_loss(model(batch), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    state = {
        "input_dim": normal_features.shape[1],
        "weights": {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
    }
    return state, actual_device


def reconstruction_errors(
    features: np.ndarray, state: dict[str, Any], *, device: str = "cpu", batch_size: int = 2048
) -> np.ndarray:
    """Return mean squared reconstruction errors without retaining GPU batches."""
    model = load_autoencoder(state, device=device)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            errors = torch.mean((model(batch) - batch).square(), dim=1)
            chunks.append(errors.cpu().numpy())
    return np.concatenate(chunks).astype(np.float64) if chunks else np.empty(0, dtype=np.float64)


def load_autoencoder(state: dict[str, Any], *, device: str) -> TabularAutoencoder:
    """Load trusted NumPy weights once for low latency inference."""
    model = TabularAutoencoder(int(state["input_dim"])).to(device)
    model.load_state_dict({key: torch.as_tensor(value) for key, value in state["weights"].items()})
    model.eval()
    return model


def tree_shap_evidence(
    model: XGBClassifier,
    row: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    class_index: int,
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    """Return exact native TreeSHAP contributions to the selected raw margin.

    These are margin contributions, not probability increments or causal effects.
    XGBoost computes the TreeSHAP values with ``pred_contribs=True``.
    """
    values = np.asarray(row, dtype=np.float32).reshape(1, -1)
    contributions = model.get_booster().predict(DMatrix(values), pred_contribs=True)
    if contributions.ndim == 3:
        class_contributions = contributions[0, class_index]
    else:
        class_contributions = contributions[0]
        if class_index == 0:
            class_contributions = -class_contributions
    ranked = np.argsort(-np.abs(class_contributions[:-1]))[:top_k]
    return [
        {
            "feature": feature_names[index],
            "value": float(values[0, index]),
            "contribution_margin": float(class_contributions[index]),
            "direction": "raises_class_margin" if class_contributions[index] >= 0 else "lowers_class_margin",
        }
        for index in ranked
    ]


def tree_shap_explanation(
    model: XGBClassifier, row: np.ndarray, feature_names: tuple[str, ...], *,
    class_index: int, top_k: int = 8,
) -> dict[str, Any]:
    """Expose exact TreeSHAP with bias and residual so a reviewer can check the sum.

All terms are raw class margins. For binary positive-class predictions the
margin is log odds; contributions are never percentage-point risk changes.
"""
    values = np.asarray(row, dtype=np.float32).reshape(1, -1)
    if values.shape[1] != len(feature_names):
        raise ValueError("SHAP feature registry does not match the supplied row")
    matrix = DMatrix(values)
    booster = model.get_booster()
    contributions = booster.predict(matrix, pred_contribs=True)
    margins = booster.predict(matrix, output_margin=True)
    if contributions.ndim == 3:
        terms = contributions[0, class_index].astype(float)
        margin = float(margins[0, class_index])
    else:
        sign = -1 if class_index == 0 else 1
        terms = contributions[0].astype(float) * sign
        margin = float(margins.reshape(-1)[0]) * sign
    ranked = np.argsort(-np.abs(terms[:-1]))[:top_k]
    items = [{"feature": feature_names[index], "value": float(values[0, index]),
              "contribution_margin": float(terms[index]),
              "direction": "raises_class_margin" if terms[index] >= 0 else "lowers_class_margin"}
             for index in ranked]
    residual = float(terms[:-1].sum() - terms[ranked].sum())
    error = abs(float(terms.sum()) - margin)
    if error > 1e-4:
        raise RuntimeError(f"TreeSHAP additivity failed: {error}")
    return {"contributions": items, "base_margin": float(terms[-1]),
            "output_margin": margin, "omitted_contribution_margin": residual,
            "additivity_error": error, "units": "raw class margin; not probability increments"}
