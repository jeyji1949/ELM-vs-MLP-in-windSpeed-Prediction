from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

y_test = np.load(BASE_DIR / "elm_y_test.npy")
y_pred_e = np.load(BASE_DIR / "elm_y_pred.npy")
y_pred_m = np.load(BASE_DIR / "mlp_y_pred.npy")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_pred, label, color in [
    (ax1, y_pred_e, "ELM", "darkorange"),
    (ax2, y_pred_m, "MLP", "crimson")
]:
    ax.scatter(y_test, y_pred, alpha=0.3, s=5, color=color)

    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)

    corr = np.corrcoef(y_test, y_pred)[0, 1]
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    r_squared_corr = corr ** 2

    print("=" * 60)
    print(f"Correlation details - {label}")
    print("=" * 60)
    print(f"Pearson correlation (r) : {corr:.4f}")
    print(f"Coefficient of determination from r (r^2) : {r_squared_corr:.4f}")
    print(f"Regression line         : y = {slope:.4f}x + {intercept:.4f}")

    ax.text(
        0.05,
        0.95,
        f"r = {corr:.4f}\nr^2 = {r_squared_corr:.4f}\ny = {slope:.3f}x + {intercept:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    ax.set_xlabel("Valeurs désirées")
    ax.set_ylabel("Valeurs prédites")
    ax.set_title(f"{label} – Désirées vs Prédites")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / "scatter_plots.png", dpi=150)
plt.show()
