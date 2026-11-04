import os

import matplotlib
import pandas as pd

# Save files only. This avoids a popup window when the script is run in the terminal.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Drawn from negative to positive so the bars read in score order.
SENTIMENT = [
    (0, "negative", "tab:red"),
    (-1, "neutral", "tab:gray"),
    (1, "positive", "tab:green"),
]


def load_clean():
    path = os.path.join(config.PROCESSED_DIR, "reviews_clean.csv")
    if not os.path.exists(path):
        raise SystemExit("Missing " + path + ". Run: python src/data_prep.py")

    print("loading", path)
    cols = ["ProductId", "Score", "Time", "review_text", "sentiment"]
    df = pd.read_csv(path, usecols=cols)
    print("rows:", len(df))
    print("charts use all cleaned reviews, including 3-star ones")
    return df


def save_fig(fig, name):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_scores(df):
    counts = df["Score"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(int).astype(str), counts.values, color="steelblue")
    ax.set_title("Score distribution")
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Number of reviews")
    save_fig(fig, "score_distribution.png")

    top_score = int(counts.idxmax())
    share = 100 * counts.max() / counts.sum()
    print("most common score:", top_score, "stars,", round(share, 1), "% of reviews")


def plot_sentiment(df):
    counts = df["sentiment"].value_counts()
    labels = [name for _, name, _ in SENTIMENT]
    values = [int(counts.get(label, 0)) for label, _, _ in SENTIMENT]
    colors = [color for _, _, color in SENTIMENT]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=colors)
    ax.set_title("Sentiment distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of reviews")
    save_fig(fig, "sentiment_distribution.png")

    total = sum(values)
    for name, n in zip(labels, values):
        print(" ", name + ":", n, "(" + str(round(100 * n / total, 1)) + "%)")


def plot_lengths(df):
    words = df["review_text"].fillna("").str.split().str.len()
    # A handful of very long reviews would squash everyone else against the y-axis.
    cap = int(words.quantile(0.99))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, name, color in SENTIMENT:
        part = words[df["sentiment"] == label]
        ax.hist(part[part <= cap], bins=40, alpha=0.45, label=name, color=color)
        print("median words,", name + ":", int(part.median()))

    ax.set_title("Review length by sentiment")
    ax.set_xlabel("Word count (up to the 99th percentile, " + str(cap) + " words)")
    ax.set_ylabel("Number of reviews")
    ax.legend()
    save_fig(fig, "review_length_by_sentiment.png")


def plot_years(df):
    years = pd.to_datetime(df["Time"], unit="s").dt.year
    counts = years.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_title("Reviews per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of reviews")
    save_fig(fig, "reviews_per_year.png")

    busiest = int(counts.idxmax())
    print("year range:", int(counts.index.min()), "to", int(counts.index.max()))
    print("most reviews in", busiest)


def plot_top_products(df):
    top = df["ProductId"].value_counts().head(20)
    ordered = top.sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(ordered.index.astype(str), ordered.values, color="steelblue")
    ax.set_title("Top 20 products by number of reviews")
    ax.set_xlabel("Number of reviews")
    ax.set_ylabel("ProductId")
    save_fig(fig, "top_products.png")

    print("most reviewed product:", top.index[0], "with", int(top.iloc[0]), "reviews")


def main():
    df = load_clean()
    plot_scores(df)
    plot_sentiment(df)
    plot_lengths(df)
    plot_years(df)
    plot_top_products(df)


if __name__ == "__main__":
    main()
