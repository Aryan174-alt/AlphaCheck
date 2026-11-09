"""A5 AI Lab: Explainability (SHAP & Deep Sequence Attention/Gradient Attribution)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def compute_deep_gradient_attribution(
    model: torch.nn.Module,
    sequence_tensor: torch.Tensor,  # shape [1, seq_len, input_dim]
    feature_names: List[str],
    output_fig: str = "reports/figures/deep_attribution.png",
) -> pd.DataFrame:
    """Compute Saliency / Gradient x Input attribution across time steps and features."""
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    model.eval()

    inputs = sequence_tensor.clone().detach().requires_grad_(True)
    output = model(inputs)

    # Compute gradients with respect to output prediction
    output.backward()
    grads = inputs.grad.squeeze(0).abs().cpu().numpy()  # [seq_len, input_dim]

    mean_per_feature = grads.mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "gradient_importance": mean_per_feature})
    df = df.sort_values("gradient_importance", ascending=False)

    plt.figure(figsize=(8, 4))
    plt.barh(df["feature"][::-1], df["gradient_importance"][::-1], color="#795290")
    plt.title("Deep Sequence Model: Feature Saliency (A5 Explainability)")
    plt.xlabel("Mean Gradient Attribution")
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    plt.close()

    return df

