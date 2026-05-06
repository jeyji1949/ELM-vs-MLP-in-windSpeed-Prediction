import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = np.loadtxt("windSpeedData.txt")

WINDOW = 10   # nombre de valeurs passées (instants 1..10)
X, y = [], []
for i in range(len(data) - WINDOW):
    X.append(data[i : i + WINDOW])   # features : t, t+1, ..., t+9
    y.append(data[i + WINDOW])       # cible     : t+10

X = np.array(X)  
y = np.array(y)   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

N      = len(X_train)  
NTest  = len(X_test)    
n      = WINDOW          
m      = 1               


L = 100   # nombre de neurones cache ils sont augmente


np.random.seed(0)
Wi = np.random.uniform(-1, 1, (L, n))   # poids entre
Bi = np.random.uniform(-1, 1, (L, 1))   # biais couche cache


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

start = time.time()

H = sigmoid(X_train @ Wi.T + Bi.T)   

# Pseudo-inverse de Moore-Penrose
Hmp  = np.linalg.pinv(H)            
beta = Hmp @ y_train                  

end = time.time()
temps_convergence = end - start


H_test       = sigmoid(X_test @ Wi.T + Bi.T)   
y_pred_elm   = H_test @ beta                   


mae  = mean_absolute_error(y_test, y_pred_elm)
mse  = mean_squared_error(y_test, y_pred_elm)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_elm)

print("=" * 50)
print("         ELM – Résultats sur les données de test")
print("=" * 50)
print(f"  Neurones cachés (L)   : {L}")
print(f"  Temps de convergence  : {temps_convergence:.4f} s")
print(f"  MAE                   : {mae:.4f}")
print(f"  MSE                   : {mse:.4f}")
print(f"  RMSE                  : {rmse:.4f}")
print(f"  R²                    : {r2:.4f}")
print("=" * 50)


np.save("elm_y_test.npy",     y_test)
np.save("elm_y_pred.npy",     y_pred_elm)
np.save("elm_metrics.npy",    np.array([mae, mse, rmse, r2, temps_convergence]))


N_plot = 200   

plt.figure(figsize=(14, 5))
plt.plot(y_test[:N_plot],     label="Valeurs désirées",    color="steelblue",  linewidth=1.5)
plt.plot(y_pred_elm[:N_plot], label="Prédictions ELM",     color="darkorange", linewidth=1.2, linestyle="--")
plt.xlabel("Échantillons de test")
plt.ylabel("Vitesse du vent (m/s)")
plt.title("ELM – Valeurs désirées vs Prédites")
plt.legend()
plt.tight_layout()
plt.savefig("elm_prediction_curve.png", dpi=150)
plt.show()
print("Courbe ELM sauvegardée : elm_prediction_curve.png")
