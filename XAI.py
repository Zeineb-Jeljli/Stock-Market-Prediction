import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
ticker          = "AAPL"
SEQUENCE_LENGTH = 60
HIDDEN_SIZE     = 50
NUM_LAYERS      = 2
DROPOUT         = 0.2
# ================================================

# -------- Rechargement données --------
data   = pd.read_csv(f"{ticker}_historical.csv", skiprows=3,
                     names=["Date","Close","Volume"],
                     index_col="Date", parse_dates=True)
prices = data["Close"].values.reshape(-1, 1)
scaler = joblib.load(f"{ticker}_scaler.pkl")
scaled = scaler.transform(prices)

X, y = [], []
for i in range(len(scaled) - SEQUENCE_LENGTH):
    X.append(scaled[i:i+SEQUENCE_LENGTH])
    y.append(scaled[i+SEQUENCE_LENGTH])
X     = np.array(X)
split = int(0.8 * len(X))
X_test = X[split:]

# -------- Rechargement modèle --------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel(1, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
model.load_state_dict(torch.load(f"{ticker}_lstm.pth"))
model.eval()

# -------- Wrapper pour SHAP --------
# SHAP a besoin d'une fonction qui prend un numpy array et retourne un numpy array
def model_predict(X_np):
    X_tensor = torch.from_numpy(X_np.reshape(-1, SEQUENCE_LENGTH, 1)).float()
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    return preds.flatten()

# -------- Calcul SHAP --------
print("Calcul des valeurs SHAP (peut prendre 1-2 minutes)...")

# On prend 50 exemples du test set pour le background (référence)
# et 20 exemples à expliquer
background = X_test[:50].reshape(50, SEQUENCE_LENGTH)
to_explain = X_test[50:70].reshape(20, SEQUENCE_LENGTH)

explainer   = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(to_explain, nsamples=100)

print("Valeurs SHAP calculées !")

# -------- Graphique 1 : Summary plot --------
# Quels jours (features) influencent le plus globalement
feature_names = [f"J-{SEQUENCE_LENGTH - i}" for i in range(SEQUENCE_LENGTH)]

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    to_explain,
    feature_names=feature_names,
    max_display=20,       # on montre les 20 jours les plus importants
    show=False,
    plot_type="bar"
)
plt.title(f"SHAP — Les 20 jours les plus influents sur la prédiction {ticker}")
plt.tight_layout()
plt.savefig(f"{ticker}_shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Graphique 1 sauvegardé : {ticker}_shap_summary.png")

# -------- Graphique 2 : Waterfall pour 1 prédiction --------
# On explique la première prédiction du test set
print("\nExplication de la prédiction #1 du test set :")

shap_exp = shap.Explanation(
    values        = shap_values[0],
    base_values   = explainer.expected_value,
    data          = to_explain[0],
    feature_names = feature_names
)

plt.figure(figsize=(12, 6))
shap.waterfall_plot(shap_exp, max_display=15, show=False)
plt.title(f"SHAP Waterfall — Décomposition d'une prédiction {ticker}")
plt.tight_layout()
plt.savefig(f"{ticker}_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Graphique 2 sauvegardé : {ticker}_shap_waterfall.png")

# -------- Résumé texte --------
importances  = np.abs(shap_values).mean(axis=0)
top5_indices = np.argsort(importances)[::-1][:5]

print("\nTop 5 jours les plus influents :")
for rank, idx in enumerate(top5_indices, 1):
    print(f"  {rank}. {feature_names[idx]} — importance : {importances[idx]:.5f}")
