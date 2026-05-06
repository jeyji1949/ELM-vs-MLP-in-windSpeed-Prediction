import numpy as np
import matplotlib.pyplot as plt


y_test   = np.load("elm_y_test.npy")
y_pred_e = np.load("elm_y_pred.npy")   # prédictions ELM
y_pred_m = np.load("mlp_y_pred.npy")   # prédictions MLP
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_pred, label, color in [
    (ax1, y_pred_e, "ELM", "darkorange"),
    (ax2, y_pred_m, "MLP", "crimson")
]:
    
    ax.scatter(y_test, y_pred, alpha=0.3, s=5, color=color)
    
   
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)  # ligne parfaite
    
    ax.set_xlabel("Valeurs désirées")
    ax.set_ylabel("Valeurs prédites")
    ax.set_title(f"{label} – Désirées vs Prédites")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("scatter_plots.png", dpi=150)
plt.show()