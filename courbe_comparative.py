import numpy as np
import matplotlib.pyplot as plt


y_test   = np.load("elm_y_test.npy")
y_pred_e = np.load("elm_y_pred.npy")   # prédictions ELM
y_pred_m = np.load("mlp_y_pred.npy")   # prédictions MLP

N = 200  

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