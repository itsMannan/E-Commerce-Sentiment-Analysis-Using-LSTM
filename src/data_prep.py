import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

random.seed(config.SEED)
np.random.seed(config.SEED)

LABEL_NAMES = {0: "negative", 1: "positive", -1: "neutral"}


def clean_text(text):
    # Leave words like "not" and "never" in. Dropping stopwords would
    # turn "not good" into "good" and the label would not match the text.
    text = text.str.lower()
    text = text.str.replace(r"<[^>]+>", " ", regex=True)
    text = text.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    return text.str.strip()


def load_raw():
    if not os.path.exists(config.RAW_PATH):
        raise SystemExit("Could not find " + config.RAW_PATH + ". Put Reviews.csv in data/raw/.")

    cols = ["ProductId", "UserId", "ProfileName", "Score", "Time", "Summary", "Text"]
    print("loading", config.RAW_PATH)
    df = pd.read_csv(config.RAW_PATH, usecols=cols)
    print("rows loaded:", len(df))
    return df


def clean_reviews(df):
    n = len(df)
    # Same review is stored once per product, so it shows up many times.
    df = df.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"])
    print("dropped duplicate reviews:", n - len(df))

    df = df.dropna(subset=["Text"]).copy()
    df["Summary"] = df["Summary"].fillna("")

    # Summary is usually the short opinion ("stale", "love it") and Text is
    # the detail. Joining them gives the model both.
    df["review_text"] = clean_text(df["Summary"] + " . " + df["Text"])
    df = df[df["review_text"] != ""].copy()

    df["sentiment"] = -1
    df.loc[df["Score"] <= 2, "sentiment"] = 0
    df.loc[df["Score"] >= 4, "sentiment"] = 1
    df["Score"] = df["Score"].astype(int)
    return df


def print_counts(name, df):
    print(name)
    counts = df["sentiment"].value_counts().sort_index()
    for label, n in counts.items():
        print(" ", LABEL_NAMES.get(label, label) + ":", int(n))


def balance_train(train):
    print_counts("train before balancing", train)
    n_neg = int((train["sentiment"] == 0).sum())
    n_pos = int((train["sentiment"] == 1).sum())
    per_class = min(n_neg, n_pos, config.SAMPLE_SIZE // 2)

    if per_class * 2 < config.SAMPLE_SIZE:
        print(
            "not enough negative reviews left to hit SAMPLE_SIZE.",
            "balanced train will be",
            per_class * 2,
        )

    neg = train[train["sentiment"] == 0].sample(n=per_class, random_state=config.SEED)
    pos = train[train["sentiment"] == 1].sample(n=per_class, random_state=config.SEED)
    # shuffle so a batch is not all one class
    balanced = pd.concat([neg, pos]).sample(frac=1, random_state=config.SEED)
    print_counts("train after balancing", balanced)
    return balanced.reset_index(drop=True)


def main():
    df = clean_reviews(load_raw())
    print_counts("all reviews", df)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    cols = ["ProductId", "UserId", "Score", "Time", "review_text", "sentiment"]

    neutral = df[df["sentiment"] == -1][cols]
    binary = df[df["sentiment"] != -1][cols].copy()
    print("saving cleaned files (this part is just disk, it can take a minute)")
    neutral.to_csv(os.path.join(config.PROCESSED_DIR, "neutral.csv"), index=False)
    df[cols].to_csv(os.path.join(config.PROCESSED_DIR, "reviews_clean.csv"), index=False)
    print("neutral rows saved:", len(neutral))

    # Split before balancing so val and test keep the real positive-heavy mix.
    train, test = train_test_split(
        binary,
        test_size=config.TEST_SIZE,
        stratify=binary["sentiment"],
        random_state=config.SEED,
    )
    val_frac = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train,
        test_size=val_frac,
        stratify=train["sentiment"],
        random_state=config.SEED,
    )

    print_counts("val (not balanced)", val)
    print_counts("test (not balanced)", test)
    train = balance_train(train)

    train.to_csv(os.path.join(config.PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(config.PROCESSED_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(config.PROCESSED_DIR, "test.csv"), index=False)

    print("train size:", len(train))
    print("val size:", len(val))
    print("test size:", len(test))
    print("saved files to", config.PROCESSED_DIR)


if __name__ == "__main__":
    main()
