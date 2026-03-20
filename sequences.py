import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
ticker = "AAPL"
SEQUENCE_LENGTH = 60          # 60 jours pour prédire le jour suivant
# ===============================================

print(f"🚀 Chargement des données prétraitées pour {ticker}...")

# Chargement du CSV du Jour 1
# Chargement du CSV avec format multi-en-têtes yfinance
data = pd.read_csv(
    f"{ticker}_historical.csv",
    skiprows=3,                          # saute les 3 lignes d'en-têtes
    names=['Date', 'Close', 'Volume'],   # renomme les colonnes
    index_col='Date',
    parse_dates=True
)
prices = data['Close'].values.reshape(-1, 1)

# Scaling (obligatoire pour LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Création des séquences
X, y = [], []
for i in range(len(scaled_prices) - SEQUENCE_LENGTH):
    X.append(scaled_prices[i:i+SEQUENCE_LENGTH])
    y.append(scaled_prices[i+SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print(f"✅ Séquences créées ! Shape X: {X.shape} | Shape y: {y.shape}")

# Split chronologique (80% train / 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape[0]} séquences | Test: {X_test.shape[0]} séquences")

# Conversion en tensors PyTorch
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test  = torch.from_numpy(X_test).float()
y_test  = torch.from_numpy(y_test).float()

# Dataset PyTorch personnalisé
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset  = StockDataset(X_test, y_test)

# DataLoaders (batch size 32)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # shuffle=False pour time series
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print("✅ DataLoaders prêts pour l'entraînement PyTorch !")

# Sauvegarde du scaler (on en aura besoin plus tard pour les prédictions)
import joblib
joblib.dump(scaler, f"{ticker}_scaler.pkl")
print(f"✅ Scaler sauvegardé : {ticker}_scaler.pkl")

# Petit graphique des données scalées (optionnel mais beau)
plt.figure(figsize=(10, 5))
plt.plot(scaled_prices, color='blue', label='Prix scalé')
plt.title(f"Prix AAPL scalé (0-1)")
plt.legend()
plt.show()