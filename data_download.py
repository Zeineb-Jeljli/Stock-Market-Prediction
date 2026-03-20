import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== CHOISIS TON STOCK ICI ====================
ticker = "AAPL"          # ← Tu peux mettre "TSLA", "NVDA", "GOOGL"...
# ============================================================

print(f"🚀 Téléchargement des données {ticker} depuis 2015...")

data = yf.download(
    ticker,
    start="2015-01-01",
    end=datetime.today().strftime('%Y-%m-%d')
)

data = data[['Close', 'Volume']].copy()

print(f"✅ Données chargées ! {data.shape[0]} jours")
print("\nDernières 5 lignes :")
print(data.tail())

# Graphique (s'ouvre automatiquement)
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], color='blue', linewidth=2)
plt.title(f"Prix de clôture de {ticker} (2015 → Aujourd'hui)")
plt.ylabel("Prix en USD")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Sauvegarde CSV (important pour les jours suivants)
data.to_csv(f"{ticker}_historical.csv", index=True)
print(f"\n✅ Fichier sauvegardé : {ticker}_historical.csv")
print("🎉 Jour 1 terminé !")