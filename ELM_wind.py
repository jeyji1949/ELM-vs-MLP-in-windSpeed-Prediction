from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt

data = np.loadtxt("windSpeedData.txt")

WINDOW = 10
X, y = [], []
for i in range(len(data) - WINDOW):
    X.append(data[i : i + WINDOW])
    y.append(data[i + WINDOW])
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

L = 400
N     = len(X_train_sc)
NTest = len(X_test_sc)
n     = WINDOW
m     = 1

rd.seed(0)
Wi = np.array([[rd.uniform(0,1) for j in range(n)] for i in range(L)])
Bi = np.array([[rd.uniform(0,1)] for i in range(L)])

start = time.time()

H     = 1.0 / (1.0 + np.exp(-(X_train_sc @ Wi.T + Bi.T)))
Hmp   = np.linalg.pinv(H)
beta  = np.dot(Hmp, y_train)

end = time.time()
temp_convergence = end - start

HTest      = 1.0 / (1.0 + np.exp(-(X_test_sc @ Wi.T + Bi.T)))
y_pred_elm = np.dot(HTest, beta)

for z in range(NTest):
    err = y_test[z] - y_pred_elm[z]
    print(y_test[z], "---", round(y_pred_elm[z], 3), "---- erreur =", round(err, 3))

mae  = mean_absolute_error(y_test, y_pred_elm)
mse  = mean_squared_error(y_test, y_pred_elm)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_elm)

print("temps de convergence :", temp_convergence)
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)

np.save("elm_y_test.npy",  y_test)
np.save("elm_y_pred.npy",  y_pred_elm)
np.save("elm_metrics.npy", np.array([mae, mse, rmse, r2, temp_convergence]))

N_plot = 400
plt.figure(figsize=(14, 5))
plt.plot(y_test[:N_plot],     label="Valeurs désirées", color="steelblue",  lw=1.8)
plt.plot(y_pred_elm[:N_plot], label="Prédictions ELM",  color="darkorange", lw=1.2, ls="--")
plt.xlabel("Échantillons de test")
plt.ylabel("Vitesse du vent (m/s)")
plt.title("ELM – Valeurs désirées vs Prédites")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elm_prediction_curve.png", dpi=150)
plt.show()