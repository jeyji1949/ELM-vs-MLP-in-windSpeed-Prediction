from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
WINDOW = 10

data = np.loadtxt(BASE_DIR / "windSpeedData.txt")
y_test = np.load(BASE_DIR / "elm_y_test.npy")
y_pred_e = np.load(BASE_DIR / "elm_y_pred.npy")
y_pred_m = np.load(BASE_DIR / "mlp_y_pred.npy")
elm_metrics = np.load(BASE_DIR / "elm_metrics.npy")
mlp_metrics = np.load(BASE_DIR / "mlp_metrics.npy")

total_samples = len(data) - WINDOW
total_test = len(y_test)
total_train = total_samples - total_test

elm_mae, elm_mse, elm_rmse, elm_r2, elm_time = elm_metrics[:5]
mlp_mae, mlp_mse, mlp_rmse, mlp_r2, mlp_time = mlp_metrics[:5]
mlp_iters = int(mlp_metrics[5]) if len(mlp_metrics) > 5 else None

print("=" * 70)
print("COMPARAISON ELM VS MLP")
print("=" * 70)
print(f"Total des mesures brutes      : {len(data)}")
print(f"Total des echantillons crees  : {total_samples}")
print(f"Total d'entrainement          : {total_train}")
print(f"Total de test                 : {total_test}")
print("=" * 70)
print(f"{'Metrique':<20}{'ELM':>15}{'MLP':>15}")
print("-" * 70)
print(f"{'MAE':<20}{elm_mae:>15.4f}{mlp_mae:>15.4f}")
print(f"{'MSE':<20}{elm_mse:>15.4f}{mlp_mse:>15.4f}")
print(f"{'RMSE':<20}{elm_rmse:>15.4f}{mlp_rmse:>15.4f}")
print(f"{'R2':<20}{elm_r2:>15.4f}{mlp_r2:>15.4f}")
print(f"{'Temps (s)':<20}{elm_time:>15.4f}{mlp_time:>15.4f}")
if mlp_iters is not None:
    print(f"{'Iterations MLP':<20}{'-':>15}{mlp_iters:>15d}")
print("=" * 70)

N = 400  

#ELM
plt.figure(figsize=(14, 5))
plt.plot(y_test[:N],    label="Valeurs désirées", color="steelblue",  lw=1.8)
plt.plot(y_pred_e[:N],  label="ELM",              color="darkorange", lw=1.2, ls="--")
plt.title("ELM – Valeurs désirées vs Prédites")
plt.xlabel("Échantillons"); plt.ylabel("Vitesse du vent (m/s)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("elm_prediction_curve.png", dpi=150)
plt.show()

#MLP
plt.figure(figsize=(14, 5))
plt.plot(y_test[:N],    label="Valeurs désirées", color="steelblue", lw=1.8)
plt.plot(y_pred_m[:N],  label="MLP",              color="crimson",   lw=1.2, ls="--")
plt.title("MLP – Valeurs désirées vs Prédites")
plt.xlabel("Échantillons"); plt.ylabel("Vitesse du vent (m/s)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("mlp_prediction_curve.png", dpi=150)
plt.show()

#ELM vs MLP
plt.figure(figsize=(14, 5))
plt.plot(y_test[:N],    label="Valeurs désirées", color="steelblue",  lw=1.8)
plt.plot(y_pred_e[:N],  label="ELM",              color="darkorange", lw=1.2, ls="--")
plt.plot(y_pred_m[:N],  label="MLP",              color="crimson",    lw=1.2, ls=":")
plt.title("Comparaison ELM vs MLP")
plt.xlabel("Échantillons"); plt.ylabel("Vitesse du vent (m/s)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("comparative_curves.png", dpi=150)
plt.show()
