import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==================== CONFIG ====================
ticker = "AAPL"
# ================================================

analyzer = SentimentIntensityAnalyzer()

# -------- Récupération des news --------
print(f"Récupération des news récentes pour {ticker}...")
ticker_obj = yf.Ticker(ticker)
news       = ticker_obj.news

if not news:
    print("Aucune news trouvée.")
else:
    print(f"{len(news)} news trouvées\n")

# -------- Analyse de sentiment --------
results = []

for article in news:
    # extraction du titre selon la structure yfinance
    content = article.get("content", {})
    title   = content.get("title", "") if isinstance(content, dict) else ""

    if not title:
        continue

    scores  = analyzer.polarity_scores(title)
    compound = scores["compound"]

    # classification
    if compound >= 0.2:
        sentiment = "POSITIF"
        color     = "green"
    elif compound <= -0.2:
        sentiment = "NEGATIF"
        color     = "red"
    else:
        sentiment = "NEUTRE"
        color     = "gray"

    results.append({
        "titre"    : title[:80],   # on tronque pour l'affichage
        "compound" : compound,
        "sentiment": sentiment,
        "color"    : color
    })

    print(f"[{sentiment:8s}] {compound:+.3f} | {title[:70]}")

# -------- Score moyen global --------
if results:
    df           = pd.DataFrame(results)
    score_moyen  = df["compound"].mean()
    nb_positifs  = len(df[df["sentiment"] == "POSITIF"])
    nb_negatifs  = len(df[df["sentiment"] == "NEGATIF"])
    nb_neutres   = len(df[df["sentiment"] == "NEUTRE"])

    print(f"\nScore sentiment moyen : {score_moyen:+.3f}")
    print(f"Positifs : {nb_positifs} | Neutres : {nb_neutres} | Négatifs : {nb_negatifs}")

    # -------- Signal marché --------
    print("\n--- Signal marché ---")
    if score_moyen >= 0.2:
        print(f"SIGNAL POSITIF ({score_moyen:+.3f}) → sentiment haussier sur {ticker}")
    elif score_moyen <= -0.2:
        print(f"SIGNAL NEGATIF ({score_moyen:+.3f}) → sentiment baissier sur {ticker}")
    else:
        print(f"SIGNAL NEUTRE ({score_moyen:+.3f}) → pas de signal clair")

    # -------- Graphique --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart des scores
    colors = df["color"].tolist()
    ax1.barh(range(len(df)), df["compound"], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels([t[:45] + "..." if len(t) > 45
                         else t for t in df["titre"]], fontsize=8)
    ax1.axvline(x=0,    color="black", linewidth=0.8, linestyle="--")
    ax1.axvline(x=0.2,  color="green", linewidth=0.8, linestyle=":")
    ax1.axvline(x=-0.2, color="red",   linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Score compound VADER")
    ax1.set_title(f"Sentiment des news {ticker}")
    ax1.set_xlim(-1, 1)

    # Pie chart
    sizes  = [nb_positifs, nb_neutres, nb_negatifs]
    labels = [f"Positif ({nb_positifs})",
              f"Neutre ({nb_neutres})",
              f"Négatif ({nb_negatifs})"]
    colors_pie = ["#2ecc71", "#95a5a6", "#e74c3c"]
    # on filtre les zéros pour éviter les erreurs matplotlib
    filtered = [(s, l, c) for s, l, c in
                zip(sizes, labels, colors_pie) if s > 0]
    if filtered:
        s, l, c = zip(*filtered)
        ax2.pie(s, labels=l, colors=c, autopct="%1.0f%%", startangle=90)
    ax2.set_title(f"Distribution sentiment {ticker}")

    plt.suptitle(f"Analyse sentiment news {ticker} — Score moyen : {score_moyen:+.3f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{ticker}_sentiment.png", dpi=150)
    plt.show()
    print(f"\nGraphique sauvegardé : {ticker}_sentiment.png")

    # -------- Sauvegarde pour Jour 6 --------
    df.to_csv(f"{ticker}_sentiment.csv", index=False)
    print(f"Données sauvegardées : {ticker}_sentiment.csv")