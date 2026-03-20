import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib

# ==================== CONFIG ====================
ticker        = "AAPL"
SEQUENCE_LENGTH = 60
HIDDEN_SIZE   = 50
NUM_LAYERS    = 2
DROPOUT       = 0.2
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.001
# ================================================


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
X_train = torch.from_numpy(X[:split]).float()
y_train = torch.from_numpy(y[:split]).float()
X_test  = torch.from_numpy(X[split:]).float()
y_test  = torch.from_numpy(y[split:]).float()

class StockDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(StockDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=False)

# -------- Définition du modèle LSTM --------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # out: (batch, seq, hidden)
        out    = out[:, -1, :]     # on prend SEULEMENT le dernier pas de temps
        return self.fc(out)        # prédiction finale : 1 valeur

model     = LSTMModel(1, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)
print(f"\nNombre de paramètres : {sum(p.numel() for p in model.parameters()):,}")

# -------- Boucle d'entraînement --------
train_losses = []

print(f"\nDébut de l'entraînement ({EPOCHS} époques)...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()          # 1. remet les gradients à zéro
        predictions = model(X_batch)   # 2. prédiction
        loss = criterion(predictions, y_batch)  # 3. calcul erreur
        loss.backward()                # 4. rétropropagation
        optimizer.step()               # 5. mise à jour des poids
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Époque {epoch+1:2d}/{EPOCHS} — Loss: {avg_loss:.6f}")

print("\nEntraînement terminé !")

# -------- Graphique de la loss --------
plt.figure(figsize=(10, 4))
plt.plot(train_losses, color="purple", linewidth=2)
plt.title("Courbe d'apprentissage — Loss par époque")
plt.xlabel("Époque")
plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------- Sauvegarde du modèle --------
torch.save(model.state_dict(), f"{ticker}_lstm.pth")
print(f"Modèle sauvegardé : {ticker}_lstm.pth")
