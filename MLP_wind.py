import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()


mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),   # 2 couches cachées
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
)

start = time.time()
mlp.fit(X_train_sc, y_train_sc)
end = time.time()
temps_convergence = end - start


y_pred_sc  = mlp.predict(X_test_sc)
y_pred_mlp = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()


mae  = mean_absolute_error(y_test, y_pred_mlp)
mse  = mean_squared_error(y_test, y_pred_mlp)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_mlp)

print("=" * 50)
print("         MLP – Résultats sur les données de test")
print("=" * 50)
print(f"  Architecture          : (100, 50)")
print(f"  Activation            : ReLU")
print(f"  Itérations            : {mlp.n_iter_}")
print(f"  Temps de convergence  : {temps_convergence:.4f} s")
print(f"  MAE                   : {mae:.4f}")
print(f"  MSE                   : {mse:.4f}")
print(f"  RMSE                  : {rmse:.4f}")
print(f"  R²                    : {r2:.4f}")
print("=" * 50)


np.save("mlp_y_test.npy",  y_test)
np.save("mlp_y_pred.npy",  y_pred_mlp)
np.save("mlp_metrics.npy", np.array([mae, mse, rmse, r2, temps_convergence, mlp.n_iter_]))


N_plot = 200

plt.figure(figsize=(14, 5))
plt.plot(y_test[:N_plot],     label="Valeurs désirées",   color="steelblue",  linewidth=1.5)
plt.plot(y_pred_mlp[:N_plot], label="Prédictions MLP",    color="crimson",    linewidth=1.2, linestyle="--")
plt.xlabel("Échantillons de test")
plt.ylabel("Vitesse du vent (m/s)")
plt.title("MLP – Valeurs désirées vs Prédites")
plt.legend()
plt.tight_layout()
plt.savefig("mlp_prediction_curve.png", dpi=150)
plt.show()
print("Courbe MLP sauvegardée : mlp_prediction_curve.png")


print("\n=== Comparaison des architectures ===")
architectures = [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]
for arch in architectures:
    m = MLPRegressor(hidden_layer_sizes=arch, activation="relu",
                     learning_rate_init=0.001, max_iter=500,
                     early_stopping=True, random_state=42)
    m.fit(X_train_sc, y_train_sc)
    yp = scaler_y.inverse_transform(m.predict(X_test_sc).reshape(-1,1)).ravel()
    print(f"  {str(arch):20s}  R²={r2_score(y_test,yp):.4f}  RMSE={np.sqrt(mean_squared_error(y_test,yp)):.4f}")

print("\n=== Comparaison des fonctions d'activation ===")
for act in ["relu", "tanh", "logistic"]:
    m = MLPRegressor(hidden_layer_sizes=(100,50), activation=act,
                     learning_rate_init=0.001, max_iter=500,
                     early_stopping=True, random_state=42)
    m.fit(X_train_sc, y_train_sc)
    yp = scaler_y.inverse_transform(m.predict(X_test_sc).reshape(-1,1)).ravel()
    print(f"  {act:10s}  R²={r2_score(y_test,yp):.4f}  RMSE={np.sqrt(mean_squared_error(y_test,yp)):.4f}")

print("\n=== Comparaison des taux d'apprentissage ===")
for lr in [0.1, 0.01, 0.001, 0.0001]:
    m = MLPRegressor(hidden_layer_sizes=(100,50), activation="relu",
                     learning_rate_init=lr, max_iter=500,
                     early_stopping=True, random_state=42)
    m.fit(X_train_sc, y_train_sc)
    yp = scaler_y.inverse_transform(m.predict(X_test_sc).reshape(-1,1)).ravel()
    print(f"  lr={lr:<8}  R²={r2_score(y_test,yp):.4f}  RMSE={np.sqrt(mean_squared_error(y_test,yp)):.4f}")
