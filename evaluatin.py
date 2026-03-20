import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# ==================== CONFIG ====================
ticker          = "AAPL"
SEQUENCE_LENGTH = 60
HIDDEN_SIZE     = 50
NUM_LAYERS      = 2
DROPOUT         = 0.2
# ================================================

# -------- Rechargement données + scaler --------
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
X, y  = np.array(X), np.array(y)
split = int(0.8 * len(X))

X_test = torch.from_numpy(X[split:]).float()
y_test = y[split:]

# -------- Rechargement du modèle --------
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
model.eval()  # mode évaluation — désactive le dropout

# -------- Prédictions sur le test set --------
with torch.no_grad():  # pas besoin de gradients en évaluation
    predictions_scaled = model(X_test).numpy()

# Dé-normalisation → vrais prix en dollars
predictions = scaler.inverse_transform(predictions_scaled)
actuals     = scaler.inverse_transform(y_test)

# -------- Métriques --------
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae  = mean_absolute_error(actuals, predictions)
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print(f"RMSE : {rmse:.2f}$")
print(f"MAE  : {mae:.2f}$")
print(f"MAPE : {mape:.2f}%")

# -------- Graphique Actual vs Predicted --------
plt.figure(figsize=(14, 6))
plt.plot(actuals,     label="Prix réel",    color="blue",   linewidth=1.5)
plt.plot(predictions, label="Prédictions",  color="orange", linewidth=1.5, alpha=0.8)
plt.title(f"{ticker} — Prix réel vs Prédictions LSTM (Test set)")
plt.xlabel("Jours")
plt.ylabel("Prix ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{ticker}_predictions.png", dpi=150)
plt.show()
print(f"Graphique sauvegardé : {ticker}_predictions.png")