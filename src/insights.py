import os
import sys

import matplotlib

# Agg is for the script. The dashboard imports this file after Streamlit, which
# already picked a backend, so leave that one alone.
if "streamlit" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

import config

# Hide a bar when the group is tiny. A percent from 8 reviews looks more precise than it is.
MIN_N = 50
TOP_N = 15

# First match wins. There is no category column in the file, so this is a guess.
# "other" is whatever is left.
CATEGORY_RULES = [
    ("baby food", r"\b(?:baby food|infant|toddler|baby formula|formula)\b"),
    ("pet food", r"\b(?:dog food|cat food|puppy|puppies|kitten|kittens|dogs?|cats?)\b"),
    ("coffee", r"\b(?:coffee|espresso|keurig|k-?cups?)\b"),
    ("tea", r"\b(?:teas?|chai|matcha)\b"),
    ("chocolate", r"\b(?:chocolate|cocoa)\b"),
    ("pasta", r"\b(?:pasta|spaghetti|noodles?|macaroni)\b"),
    ("sauces", r"\b(?:sauces?|salsa|dressing|marinade|ketchup|mustard)\b"),
    ("snacks", r"\b(?:chips?|cookies?|crackers?|snacks?|popcorn|pretzels?|candy|granola)\b"),
]
CATEGORY_ORDER = [name for name, _ in CATEGORY_RULES] + ["other"]

# Count charts only. "not" stays in, otherwise "not good" disappears from the bars.
STOP_WORDS = list(ENGLISH_STOP_WORDS - {"not", "no", "never", "nor"})

THEMES = [
    ("stale", r"\bstale\b"),
    ("expired", r"\bexpir\w*\b"),
    ("packaging", r"\bpackag\w*\b"),
    ("damaged", r"\bdamag\w*\b"),
    ("late/shipping", r"\b(?:late|shipping|shipped|shipment|delivery|delivered)\b"),
    ("price", r"\b(?:price|priced|expensive|overpriced)\b"),
    ("taste", r"\b(?:taste|tastes|tasted|tasting|flavor|flavour)\b"),
    ("smell", r"\b(?:smell|smells|smelled|aroma|odor|odour)\b"),
    ("broken/open", r"\b(?:broken|opened|unsealed|torn)\b|\bopen bag\b"),
    ("small portion", r"\b(?:small portion|too small|tiny portion|few pieces|mostly air|half full)\b"),
]


def assign_category(text):
    # "hot dog" contains "dog". Swap it first so those reviews are not filed as pet food.
    cleaned = text.fillna("").astype(str)
    cleaned = cleaned.str.replace(r"\bhot dogs?\b", "hotdog", regex=True)
    cat = pd.Series("other", index=cleaned.index)
    for name, pattern in CATEGORY_RULES:
        hit = cleaned.str.contains(pattern, regex=True, na=False)
        cat = cat.mask((cat == "other") & hit, name)
    return cat


def load_clean():
    path = os.path.join(config.PROCESSED_DIR, "reviews_clean.csv")
    if not os.path.exists(path):
        raise SystemExit("Missing " + path + ". Run: python src/data_prep.py")
    print("loading", path)
    cols = ["ProductId", "Time", "review_text", "sentiment"]
    df = pd.read_csv(path, usecols=cols)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].astype(int)
    print("rows:", len(df))
    return df


def save_fig(fig, name):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def save_table(df, name):
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    path = os.path.join(config.REPORT_DIR, name)
    df.to_csv(path, index=False)
    print("saved", path)


def class_sample(df):
    pos = df[df["sentiment"] == 1]
    neg = df[df["sentiment"] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        raise SystemExit("Need both positive and negative reviews. Run data_prep.py first.")
    # Same count from each class so the longer positive class does not win just by size.
    print("word charts use", n, "reviews from each class")
    pos = pos.sample(n=n, random_state=config.SEED)
    neg = neg.sample(n=n, random_state=config.SEED)
    return pos, neg


def top_terms(texts, ngram):
    vec = CountVectorizer(stop_words=STOP_WORDS, ngram_range=(ngram, ngram), min_df=5)
    X = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    names = vec.get_feature_names_out()
    order = counts.argsort()[::-1][:TOP_N]
    return names[order], counts[order]


def draw_bars(ax, names, counts, title, color):
    # barh draws the first row at the bottom, so flip to put the largest on top
    ax.barh(names[::-1], counts[::-1], color=color)
    ax.set_title(title)
    ax.set_xlabel("Count")


def plot_common_words(pos, neg):
    rows = []
    for ngram, filename, title in [
        (1, "top_unigrams.png", "Most common words"),
        (2, "top_bigrams.png", "Most common phrases"),
    ]:
        print("counting", title.lower())
        pos_names, pos_counts = top_terms(pos["review_text"], ngram)
        neg_names, neg_counts = top_terms(neg["review_text"], ngram)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        draw_bars(axes[0], neg_names, neg_counts, "Negative", "tab:red")
        draw_bars(axes[1], pos_names, pos_counts, "Positive", "tab:green")
        fig.suptitle(title)
        fig.tight_layout()
        save_fig(fig, filename)
        kind = "unigram" if ngram == 1 else "bigram"
        for term, count in zip(neg_names, neg_counts):
            rows.append({"class": "negative", "kind": kind, "term": term, "count": int(count)})
        for term, count in zip(pos_names, pos_counts):
            rows.append({"class": "positive", "kind": kind, "term": term, "count": int(count)})
    save_table(pd.DataFrame(rows), "top_terms.csv")


def plot_coefficients():
    tfidf_path = os.path.join(config.MODEL_DIR, "tfidf.joblib")
    clf_path = os.path.join(config.MODEL_DIR, "logreg.joblib")
    if not (os.path.exists(tfidf_path) and os.path.exists(clf_path)):
        print("skipping coefficient chart. Run: python src/train.py")
        return

    tfidf = joblib.load(tfidf_path)
    clf = joblib.load(clf_path)
    names = np.asarray(tfidf.get_feature_names_out())
    coef = clf.coef_.ravel()
    if len(names) != len(coef):
        print("skipping coefficient chart, vectorizer and model sizes differ")
        return

    order = np.argsort(coef)
    neg_idx = order[:TOP_N]
    pos_idx = order[-TOP_N:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # barh puts the first item at the bottom, so flip to put the strongest on top
    axes[0].barh(names[neg_idx][::-1], coef[neg_idx][::-1], color="tab:red")
    axes[0].set_title("Tied to negative")
    axes[0].set_xlabel("Logistic regression coefficient")
    axes[1].barh(names[pos_idx][::-1], coef[pos_idx][::-1], color="tab:green")
    axes[1].set_title("Tied to positive")
    axes[1].set_xlabel("Logistic regression coefficient")
    fig.suptitle("Baseline coefficients")
    fig.tight_layout()
    save_fig(fig, "logreg_coefficients.png")

    rows = []
    for kind, idx in [("negative", neg_idx), ("positive", pos_idx)]:
        for i in idx:
            rows.append({"class": kind, "term": names[i], "coefficient": float(coef[i])})
    save_table(pd.DataFrame(rows), "logreg_top_terms.csv")


def plot_themes(df):
    neg = df[df["sentiment"] == 0]
    print("theme counts use every negative review:", len(neg))
    print("a review can land in more than one theme")
    rows = []
    for name, pattern in THEMES:
        n = int(neg["review_text"].str.contains(pattern, regex=True, na=False).sum())
        rows.append({"theme": name, "n_reviews": n})
        print(" ", name + ":", n)
    themes = pd.DataFrame(rows).sort_values("n_reviews")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(themes["theme"], themes["n_reviews"], color="tab:red")
    ax.set_title("Complaint themes in negative reviews")
    ax.set_xlabel("Number of negative reviews")
    save_fig(fig, "complaint_themes.png")
    save_table(themes.sort_values("n_reviews", ascending=False), "complaint_themes.csv")


def binary_summary(df, col):
    binary = df[df["sentiment"] != -1]
    grouped = binary.groupby(col).agg(n=("sentiment", "size"), n_pos=("sentiment", "sum"))
    grouped = grouped[grouped["n"] >= MIN_N].copy()
    grouped["pct_positive"] = 100 * grouped["n_pos"] / grouped["n"]
    return grouped


def plot_categories(df):
    print("assigning keyword categories")
    df = df.copy()
    df["category"] = assign_category(df["review_text"])
    print(df["category"].value_counts().to_string())

    summary = binary_summary(df, "category")
    summary = summary.reindex([c for c in CATEGORY_ORDER if c in summary.index]).dropna()
    if summary.empty:
        print("no category had at least", MIN_N, "reviews")
        return df
    summary = summary.sort_values("pct_positive")
    labels = [idx + " (n=" + str(int(summary.loc[idx, "n"])) + ")" for idx in summary.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, summary["pct_positive"].values, color="steelblue")
    ax.set_xlabel("Percent positive")
    ax.set_xlim(0, 100)
    ax.set_title("Sentiment by keyword category")
    save_fig(fig, "sentiment_by_category.png")

    out = summary.reset_index()
    if "category" not in out.columns:
        out = out.rename(columns={out.columns[0]: "category"})
    save_table(out, "sentiment_by_category.csv")
    return df


def plot_products(df):
    summary = binary_summary(df, "ProductId")
    if summary.empty:
        print("no product had at least", MIN_N, "reviews")
        return
    summary = summary.sort_values("n", ascending=False).head(TOP_N)
    summary = summary.sort_values("pct_positive")
    labels = [idx + " (n=" + str(int(n)) + ")" for idx, n in summary["n"].items()]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, summary["pct_positive"].values, color="steelblue")
    ax.set_xlabel("Percent positive")
    ax.set_xlim(0, 100)
    ax.set_title("Sentiment for the top " + str(TOP_N) + " products")
    save_fig(fig, "sentiment_by_product.png")

    out = summary.sort_values("n", ascending=False).reset_index()
    if "ProductId" not in out.columns:
        out = out.rename(columns={out.columns[0]: "ProductId"})
    save_table(out, "sentiment_by_product.csv")


def plot_over_time(df):
    binary = df[df["sentiment"] != -1].copy()
    binary["year"] = pd.to_datetime(binary["Time"], unit="s").dt.year
    grouped = binary.groupby("year").agg(n=("sentiment", "size"), n_pos=("sentiment", "sum"))
    small = grouped[grouped["n"] < MIN_N]
    if len(small):
        print("years left out (under " + str(MIN_N) + " non-neutral reviews):", list(small.index.astype(int)))
    grouped = grouped[grouped["n"] >= MIN_N].copy()
    if grouped.empty:
        print("no year had at least", MIN_N, "reviews")
        return
    grouped["pct_positive"] = 100 * grouped["n_pos"] / grouped["n"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grouped.index.astype(int), grouped["pct_positive"], marker="o", color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percent positive")
    ax.set_ylim(0, 100)
    ax.set_title("Percent positive by year")
    save_fig(fig, "sentiment_over_time.png")

    out = grouped.reset_index()
    save_table(out, "sentiment_by_year.csv")


def main():
    df = load_clean()
    pos, neg = class_sample(df)
    plot_common_words(pos, neg)
    plot_coefficients()
    plot_themes(df)
    df = plot_categories(df)
    plot_products(df)
    plot_over_time(df)
    print("category labels are keyword guesses. A review with two foods only gets the first match.")


if __name__ == "__main__":
    main()
