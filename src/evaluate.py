import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config

# Rough word lists for the error check. Not a second model.
NEGATION_RE = r"\b(not|no|never|nor|neither)\b|n't"
MIXED_RE = r"\b(but|however|although|though)\b"
SHIP_RE = r"\b(shipping|shipped|shipment|package|packaging|packaged|damaged|damage|delivery|delivered)\b"
TASTE_RE = r"\b(taste|tastes|tasted|flavor|flavour|smell|smells|aroma)\b"
POS_RE = r"\b(?:good|great|love|loved|excellent|delicious|amazing|perfect|best|tasty)\b"
NEG_RE = r"\b(?:bad|terrible|awful|horrible|worst|stale|disgusting|disappointed|bland|gross)\b"
SARCASM_PHRASES = ["yeah right", "thanks for nothing", "what a joke"]
SHORT_WORDS = 8


def load_test():
    path = os.path.join(config.PROCESSED_DIR, "test.csv")
    if not os.path.exists(path):
        raise SystemExit("Missing " + path + ". Run: python src/data_prep.py")
    df = pd.read_csv(path, usecols=["review_text", "sentiment", "Score"])
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].astype(int)
    df["Score"] = df["Score"].astype(int)
    df["n_words"] = df["review_text"].str.split().str.len().fillna(0).astype(int)
    return df


def count_neutral():
    path = os.path.join(config.PROCESSED_DIR, "neutral.csv")
    if not os.path.exists(path):
        return None
    # Score is enough. The file can be large.
    return len(pd.read_csv(path, usecols=["Score"]))


def predict_probs(model, texts):
    print("scoring test set")
    ds = tf.data.Dataset.from_tensor_slices(texts.tolist()).batch(config.BATCH_SIZE)
    probs = model.predict(ds, verbose=1)
    return np.asarray(probs).reshape(-1)


def save_metrics(y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, probs)

    print("accuracy:", round(acc, 4))
    print("precision:", round(prec, 4))
    print("recall:", round(rec, 4))
    print("f1:", round(f1, 4))
    print("roc_auc:", round(auc, 4))
    print("precision and recall are for the positive class")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "positive"],
        digits=4,
        zero_division=0,
    )
    print(report)

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    report_path = os.path.join(config.REPORT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("saved", report_path)

    metrics = pd.DataFrame(
        [
            {
                "model": "lstm",
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
            }
        ]
    )
    metrics_path = os.path.join(config.REPORT_DIR, "metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print("saved", metrics_path)
    return acc


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    cutoff = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cutoff else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    path = os.path.join(config.FIG_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    print("confusion matrix [true neg, false pos] / [false neg, true pos]:")
    print(cm)


def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="LSTM (AUC " + str(round(auc, 3)) + ")")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    path = os.path.join(config.FIG_DIR, "roc_curve.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def save_mistakes(df):
    wrong = df[df["pred"] != df["sentiment"]].copy()
    # High probability on a negative review, or low probability on a positive one.
    confident = wrong["prob"].where(wrong["pred"] == 1, 1 - wrong["prob"])
    wrong = wrong.loc[confident.sort_values(ascending=False).index]
    out = wrong.rename(
        columns={
            "review_text": "text",
            "sentiment": "true label",
            "pred": "predicted label",
            "prob": "predicted probability",
            "Score": "original Score",
            "n_words": "review length",
        }
    )
    cols = [
        "text",
        "true label",
        "predicted label",
        "predicted probability",
        "original Score",
        "review length",
    ]
    path = os.path.join(config.REPORT_DIR, "misclassified_reviews.csv")
    out[cols].to_csv(path, index=False)
    print("misclassified:", len(out))
    print("saved", path)


def clip_text(text):
    text = " ".join(str(text).split())
    if len(text) > 200:
        return text[:200] + "..."
    return text


def print_confident_errors(df):
    fp = df[(df["sentiment"] == 0) & (df["pred"] == 1)].sort_values("prob", ascending=False)
    fn = df[(df["sentiment"] == 1) & (df["pred"] == 0)].sort_values("prob", ascending=True)
    print("false positives:", len(fp), "(labeled negative, predicted positive)")
    print("false negatives:", len(fn), "(labeled positive, predicted negative)")

    print("most confident false positives")
    if fp.empty:
        print("  none")
    for _, row in fp.head(5).iterrows():
        print("  score", int(row["Score"]), "prob", round(float(row["prob"]), 3))
        print("   ", clip_text(row["review_text"]))

    print("most confident false negatives")
    if fn.empty:
        print("  none")
    for _, row in fn.head(5).iterrows():
        print("  score", int(row["Score"]), "prob", round(float(row["prob"]), 3))
        print("   ", clip_text(row["review_text"]))


def pattern_masks(text):
    sarcasm = pd.Series(False, index=text.index)
    for phrase in SARCASM_PHRASES:
        sarcasm = sarcasm | text.str.contains(phrase, regex=False)
    return {
        "negation": text.str.contains(NEGATION_RE, regex=True),
        "mixed_but": text.str.contains(MIXED_RE, regex=True),
        "shipping_or_packaging": text.str.contains(SHIP_RE, regex=True),
        "sarcasm_phrase": sarcasm,
    }


def check_patterns(df):
    masks = pattern_masks(df["review_text"])
    masks["very_short"] = df["n_words"] <= SHORT_WORDS
    # Low stars with mostly positive words, and the reverse.
    # Skip "not good" style lines so the word "good" does not count as praise.
    pos_hits = df["review_text"].str.count(POS_RE)
    neg_hits = df["review_text"].str.count(NEG_RE)
    masks["low_star_but_positive_words"] = (
        (df["Score"] <= 2) & ~masks["negation"] & (pos_hits > neg_hits) & (pos_hits > 0)
    )
    masks["high_star_but_negative_words"] = (df["Score"] >= 4) & (neg_hits > pos_hits) & (neg_hits > 0)

    rows = []
    wrong = df["pred"] != df["sentiment"]
    print("pattern check (keyword counts, not a careful read of each review)")
    print("very short means", SHORT_WORDS, "words or fewer")
    for name, mask in masks.items():
        n_all = int(mask.sum())
        n_err = int((mask & wrong).sum())
        rate_all = n_all / len(df)
        rate_err = n_err / int(wrong.sum()) if int(wrong.sum()) else 0
        rows.append(
            {
                "pattern": name,
                "n_test": n_all,
                "rate_test": rate_all,
                "n_misclassified": n_err,
                "rate_misclassified": rate_err,
            }
        )
        print(
            " ",
            name + ":",
            n_all,
            "on the test set (" + str(round(100 * rate_all, 1)) + "%) ,",
            n_err,
            "of the mistakes (" + str(round(100 * rate_err, 1)) + "%)",
        )

    out = pd.DataFrame(rows)
    path = os.path.join(config.REPORT_DIR, "error_patterns.csv")
    out.to_csv(path, index=False)
    print("saved", path)
    return masks


def performance_tables(df):
    by_score = []
    for score, part in df.groupby("Score"):
        by_score.append(
            {
                "score": int(score),
                "n": len(part),
                "accuracy": float((part["pred"] == part["sentiment"]).mean()),
                "mean_predicted_prob": float(part["prob"].mean()),
            }
        )
    by_score = pd.DataFrame(by_score).sort_values("score")
    score_path = os.path.join(config.REPORT_DIR, "performance_by_score.csv")
    by_score.to_csv(score_path, index=False)
    print("accuracy by star rating")
    print(by_score.to_string(index=False))
    print("saved", score_path)

    bins = [0, SHORT_WORDS, 30, 80, 200, 100000]
    labels = ["1-8", "9-30", "31-80", "81-200", "201+"]
    df = df.copy()
    df["length_bin"] = pd.cut(df["n_words"], bins=bins, labels=labels)
    by_len = []
    for name, part in df.groupby("length_bin", observed=True):
        by_len.append(
            {
                "length_bin": str(name),
                "n": len(part),
                "accuracy": float((part["pred"] == part["sentiment"]).mean()),
                "n_wrong": int((part["pred"] != part["sentiment"]).sum()),
            }
        )
    by_len = pd.DataFrame(by_len)
    len_path = os.path.join(config.REPORT_DIR, "performance_by_length.csv")
    by_len.to_csv(len_path, index=False)
    print("accuracy by review length")
    print(by_len.to_string(index=False))
    print("saved", len_path)
    return by_score, by_len


def pct(part, whole):
    if not whole:
        return 0.0
    return 100 * part / whole


def write_limitations(df, masks, by_score, n_neutral):
    n = len(df)
    counts = df["Score"].value_counts().sort_index()
    count_txt = ", ".join(str(int(s)) + "-star " + str(int(c)) for s, c in counts.items())
    n5 = int((df["Score"] == 5).sum())
    acc = float((df["pred"] == df["sentiment"]).mean())

    acc_map = {int(r["score"]): float(r["accuracy"]) for _, r in by_score.iterrows()}
    n_map = {int(r["score"]): int(r["n"]) for _, r in by_score.iterrows()}
    star_bits = []
    for score in [1, 2, 4, 5]:
        if score in acc_map:
            star_bits.append(
                str(score)
                + "-star "
                + str(round(100 * acc_map[score], 1))
                + "% correct (n="
                + str(n_map[score])
                + ")"
            )

    star_line = "Correct rate by star on the test set: " + ", ".join(star_bits) + "."
    if all(s in acc_map for s in [1, 2, 4, 5]):
        harder_low = acc_map[2] < acc_map[1]
        harder_high = acc_map[4] < acc_map[5]
        if harder_low and harder_high:
            star_line += " 2-star reviews were harder than 1-star, and 4-star reviews were harder than 5-star."
        elif (not harder_low) and (not harder_high):
            star_line += " In this run, 2-star and 4-star reviews were not harder than 1-star and 5-star."
        else:
            star_line += " The 2-star vs 1-star gap and the 4-star vs 5-star gap do not point the same way here."

    neg = df["Score"] <= 2
    n_neg = int(neg.sum())
    n_ship = int((neg & masks["shipping_or_packaging"]).sum())

    exact = int(df["review_text"].duplicated(keep=False).sum())
    opening = df["review_text"].str.split().str[:15].str.join(" ")
    long_enough = df["n_words"] >= 8
    near = int(opening[long_enough].duplicated(keep=False).sum())

    taste_n = int(df["review_text"].str.contains(TASTE_RE, regex=True).sum())
    short = df["n_words"] <= SHORT_WORDS
    short_acc = float((df.loc[short, "pred"] == df.loc[short, "sentiment"]).mean()) if short.any() else 0
    other_acc = float((df.loc[~short, "pred"] == df.loc[~short, "sentiment"]).mean()) if (~short).any() else 0
    low_pos = int(masks["low_star_but_positive_words"].sum())
    high_neg = int(masks["high_star_but_negative_words"].sum())

    if n_neutral is None:
        neutral_line = "I dropped score 3 from the model. neutral.csv was not in data/processed, so I do not have that count here."
    else:
        neutral_line = (
            "I dropped score 3 from the model. A 3-star review is not clearly positive or negative. "
            + str(n_neutral)
            + " of those rows are in neutral.csv."
        )

    lines = [
        "Limitations of the sentiment labels",
        "",
        "I used the star rating as the sentiment label. That is a noisy stand-in for what the person actually felt. "
        "Someone can like the food and still leave a low score because the order arrived broken, or leave 5 stars on a review that barely says anything.",
        "",
        star_line,
        "",
        neutral_line,
        "",
        "The test set keeps the real mix of stars: "
        + count_txt
        + ". "
        + str(n5)
        + " of "
        + str(n)
        + " are 5 stars ("
        + str(round(pct(n5, n), 1))
        + "%). Test accuracy was "
        + str(round(100 * acc, 1))
        + "%. With this many 5-star reviews, accuracy is easy to overread. The other metrics are in metrics.csv.",
        "",
        "Among reviews labeled negative (1 or 2 stars), "
        + str(n_ship)
        + " of "
        + str(n_neg)
        + " ("
        + str(round(pct(n_ship, n_neg), 1))
        + "%) mention shipping, packaging, or damage. Those rows are still labeled as negative product sentiment, even when the complaint is about the order.",
        "",
        "The prep step drops a review if the same user, profile name, time, and text already appeared. On this test set, "
        + str(exact)
        + " reviews still share an exact cleaned text with another test review, and "
        + str(near)
        + " reviews of at least 8 words share the same first 15 words with another one. I would not call that a measured duplicate rate for the whole file, only what showed up here.",
        "",
        str(taste_n)
        + " test reviews ("
        + str(round(pct(taste_n, n), 1))
        + "%) mention taste, flavor, or smell. Food taste is subjective, and the label is still just the star rating, so two people can describe the same product differently and land on different scores.",
        "",
        "A crude word check found "
        + str(low_pos)
        + " low-star reviews that use more positive words than negative ones (ignoring lines with not/never), and "
        + str(high_neg)
        + " high-star reviews that use more negative words than positive ones. Some of those stars disagree with the wording. The lists are small, so this is only a hint.",
        "",
        "Reviews of "
        + str(SHORT_WORDS)
        + " words or fewer were "
        + str(round(100 * short_acc, 1))
        + "% correct (n="
        + str(int(short.sum()))
        + "). Longer reviews were "
        + str(round(100 * other_acc, 1))
        + "% correct. A short review gives the model very little text next to a strong star.",
        "",
        "The shipping, taste, and contradiction counts are keyword matches. They are not a careful reading of every review, and they do not prove why each mistake happened.",
    ]
    path = os.path.join(config.REPORT_DIR, "label_limitations.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved", path)


def main():
    model_path = os.path.join(config.MODEL_DIR, "lstm.keras")
    if not os.path.exists(model_path):
        raise SystemExit("Missing " + model_path + ". Run: python src/train.py")

    os.makedirs(config.FIG_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    df = load_test()
    print("test size:", len(df))
    model = keras.models.load_model(model_path)
    df["prob"] = predict_probs(model, df["review_text"])
    df["pred"] = (df["prob"] >= 0.5).astype(int)

    save_metrics(df["sentiment"], df["pred"], df["prob"])
    plot_confusion(df["sentiment"], df["pred"])
    plot_roc(df["sentiment"], df["prob"])
    save_mistakes(df)
    print_confident_errors(df)
    masks = check_patterns(df)
    by_score, _ = performance_tables(df)
    write_limitations(df, masks, by_score, count_neutral())


if __name__ == "__main__":
    main()
