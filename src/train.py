import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config

random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

# Stop if val loss does not improve for this many epochs.
PATIENCE = 2


def load_split(name):
    path = os.path.join(config.PROCESSED_DIR, name + ".csv")
    if not os.path.exists(path):
        raise SystemExit("Missing " + path + ". Run: python src/data_prep.py")
    df = pd.read_csv(path, usecols=["review_text", "sentiment"])
    df["review_text"] = df["review_text"].fillna("").astype(str)
    return df


def make_ds(texts, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((texts.tolist(), labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=config.SEED)
    return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_vectorizer(train_text):
    print("adapting vocabulary")
    vectorizer = layers.TextVectorization(
        max_tokens=config.MAX_WORDS,
        output_sequence_length=config.MAX_LEN,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_text.tolist()).batch(2048))
    return vectorizer


def save_vocab(vectorizer):
    vocab = vectorizer.get_vocabulary()
    path = os.path.join(config.MODEL_DIR, "vocab.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    print("vocab size:", len(vocab))
    print("saved", path)


def build_model(vectorizer):
    # Index 0 is padding. mask_zero lets the LSTM skip those steps.
    # One direction only. Bidirectional would be about twice as slow on CPU.
    inputs = keras.Input(shape=(), dtype="string")
    x = vectorizer(inputs)
    x = layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=config.EMBED_DIM,
        mask_zero=True,
    )(x)
    x = layers.LSTM(config.LSTM_UNITS)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def plot_history(history):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, hist["loss"], marker="o", label="train")
    axes[0].plot(epochs, hist["val_loss"], marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary crossentropy")
    axes[0].legend()

    axes[1].plot(epochs, hist["accuracy"], marker="o", label="train")
    axes[1].plot(epochs, hist["val_accuracy"], marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(config.FIG_DIR, "training_curves.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def score_model(name, y_true, probs):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    scores = {
        "model": name,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
    }
    print(name)
    for key, value in scores.items():
        if key == "model":
            continue
        print(" ", key + ":", round(float(value), 4))
    return scores


def fit_baseline(train_text, y_train, test_text, y_test):
    # Bigrams so a phrase like "not good" can be one feature.
    # Stopwords stay in, same as the LSTM input.
    print("fitting tf-idf + logistic regression")
    tfidf = TfidfVectorizer(max_features=config.MAX_WORDS, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    clf = LogisticRegression(max_iter=500, solver="liblinear", random_state=config.SEED)
    clf.fit(X_train, y_train)

    joblib.dump(tfidf, os.path.join(config.MODEL_DIR, "tfidf.joblib"))
    joblib.dump(clf, os.path.join(config.MODEL_DIR, "logreg.joblib"))
    print("saved baseline to", config.MODEL_DIR)
    probs = clf.predict_proba(X_test)[:, 1]
    return score_model("tfidf_logreg", y_test, probs)


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    print("train size:", len(train))
    print("val size:", len(val))
    print("test size:", len(test))
    print("train positive:", int((train["sentiment"] == 1).sum()))
    print("train negative:", int((train["sentiment"] == 0).sum()))
    print("test keeps the real mix of scores, so accuracy can look high")
    print("if this is too slow, lower EPOCHS in src/config.py")
    print("SAMPLE_SIZE only shrinks train. Change it, then rerun data_prep.py")

    y_train = train["sentiment"].to_numpy(dtype="float32")
    y_val = val["sentiment"].to_numpy(dtype="float32")
    y_test = test["sentiment"].to_numpy(dtype="int64")

    baseline_scores = fit_baseline(train["review_text"], y_train.astype(int), test["review_text"], y_test)

    vectorizer = build_vectorizer(train["review_text"])
    save_vocab(vectorizer)
    model = build_model(vectorizer)
    model.summary()

    model_path = os.path.join(config.MODEL_DIR, "lstm.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    print("training lstm")
    history = model.fit(
        make_ds(train["review_text"], y_train, shuffle=True),
        validation_data=make_ds(val["review_text"], y_val),
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
    hist_path = os.path.join(config.REPORT_DIR, "train_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print("saved", hist_path)
    print("best val loss:", round(min(history.history["val_loss"]), 4))
    plot_history(history)

    print("scoring lstm on the test set")
    probs = model.predict(make_ds(test["review_text"], y_test.astype("float32")), verbose=1)
    lstm_scores = score_model("lstm", y_test, probs)

    comparison = pd.DataFrame([lstm_scores, baseline_scores])
    comp_path = os.path.join(config.REPORT_DIR, "model_comparison.csv")
    comparison.to_csv(comp_path, index=False)
    print("saved", comp_path)


if __name__ == "__main__":
    main()
