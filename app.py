import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd
import streamlit as st

import config
from insights import CATEGORY_ORDER, assign_category

PAGE_SIZE = 20
LABELS = {0: "negative", 1: "positive", -1: "neutral"}

EXAMPLES = {
    "Good coffee": "love this coffee. smooth taste and it arrived fresh. i buy it every month.",
    "Stale snack": "the chips were stale and the bag was open when it arrived.",
    "Mixed review": "good taste but the jar arrived cracked and shipping took three weeks.",
    "Short negation": "not good",
}


def read_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_reviews(path):
    df = pd.read_csv(path, usecols=["ProductId", "Score", "review_text", "sentiment"])
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].astype(int)
    df["Score"] = df["Score"].astype(int)
    df["category"] = assign_category(df["review_text"])
    df["sentiment_name"] = df["sentiment"].map(LABELS)
    return df


@st.cache_resource
def load_lstm(path):
    # Import here so the overview page can open before TensorFlow loads.
    import keras

    return keras.models.load_model(path)


def predict_texts(model, texts):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(list(texts)).batch(32)
    probs = model.predict(ds, verbose=0)
    return probs.reshape(-1)


def show_fig(name, script):
    path = os.path.join(config.FIG_DIR, name)
    if os.path.exists(path):
        st.image(path)
    else:
        st.warning("Missing " + name + ". Run: python src/" + script)


def page_slice(df, page_key):
    n_pages = max(1, (len(df) + PAGE_SIZE - 1) // PAGE_SIZE)
    if page_key in st.session_state and st.session_state[page_key] > n_pages:
        st.session_state[page_key] = 1
    page = st.number_input("Page", min_value=1, max_value=n_pages, step=1, key=page_key)
    start = (int(page) - 1) * PAGE_SIZE
    stop = min(start + PAGE_SIZE, len(df))
    st.write("Showing", start + 1, "to", stop, "of", len(df))
    return df.iloc[start:stop]


def get_reviews():
    path = os.path.join(config.PROCESSED_DIR, "reviews_clean.csv")
    if not os.path.exists(path):
        return None
    with st.spinner("Loading reviews"):
        return load_reviews(path)


def get_model():
    path = os.path.join(config.MODEL_DIR, "lstm.keras")
    if not os.path.exists(path):
        return None
    return load_lstm(path)


def page_overview():
    st.header("Overview")
    st.write("Counts below come from the cleaned file. Model numbers come from the report files, after you train and evaluate.")

    df = get_reviews()
    if df is None:
        st.warning("Missing data/processed/reviews_clean.csv. Run: python src/data_prep.py")
    else:
        n_pos = int((df["sentiment"] == 1).sum())
        n_neg = int((df["sentiment"] == 0).sum())
        n_neu = int((df["sentiment"] == -1).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reviews", len(df))
        c2.metric("Positive", n_pos)
        c3.metric("Negative", n_neg)
        c4.metric("Neutral (score 3)", n_neu)

    metrics_path = os.path.join(config.REPORT_DIR, "metrics.csv")
    metrics = read_csv(metrics_path)
    if metrics is None:
        st.warning("Missing outputs/reports/metrics.csv. Run: python src/evaluate.py")
    else:
        st.subheader("LSTM on the test set")
        st.dataframe(metrics, hide_index=True)

    comp_path = os.path.join(config.REPORT_DIR, "model_comparison.csv")
    comp = read_csv(comp_path)
    if comp is None:
        st.warning("Missing outputs/reports/model_comparison.csv. Run: python src/train.py")
    else:
        st.subheader("LSTM vs logistic regression")
        st.dataframe(comp, hide_index=True)

    st.subheader("Charts")
    left, right = st.columns(2)
    with left:
        show_fig("score_distribution.png", "eda.py")
        show_fig("sentiment_by_category.png", "insights.py")
    with right:
        show_fig("sentiment_distribution.png", "eda.py")
        show_fig("sentiment_over_time.png", "insights.py")
    show_fig("top_products.png", "eda.py")


def page_explorer():
    st.header("Review explorer")
    st.write("Categories are keyword guesses. The model prediction is only for the rows on this page.")

    df = get_reviews()
    if df is None:
        st.warning("Missing data/processed/reviews_clean.csv. Run: python src/data_prep.py")
        return

    c1, c2, c3 = st.columns(3)
    sentiment = c1.selectbox("Sentiment", ["All"] + ["positive", "negative", "neutral"])
    category = c2.selectbox("Category", ["All"] + CATEGORY_ORDER)
    score = c3.selectbox("Score", ["All", "1", "2", "3", "4", "5"])
    product = st.text_input("Product id contains")
    keyword = st.text_input("Keyword in the review")

    view = df
    if sentiment != "All":
        view = view[view["sentiment_name"] == sentiment]
    if category != "All":
        view = view[view["category"] == category]
    if score != "All":
        view = view[view["Score"] == int(score)]
    if product.strip():
        view = view[view["ProductId"].astype(str).str.contains(product.strip(), case=False, regex=False)]
    if keyword.strip():
        view = view[view["review_text"].str.contains(keyword.strip(), case=False, regex=False)]

    if view.empty:
        st.write("No reviews match those filters.")
        return

    page = page_slice(view, "explorer_page")
    model = get_model()
    show = page.copy()
    if model is None:
        st.warning("Missing models/lstm.keras. Run: python src/train.py. The table is the review text only.")
    else:
        with st.spinner("Scoring this page"):
            probs = predict_texts(model, show["review_text"].tolist())
        show["prob_positive"] = probs
        show["model_prediction"] = ["positive" if p >= 0.5 else "negative" for p in probs]

    cols = ["review_text", "Score", "sentiment_name", "category", "ProductId"]
    if "model_prediction" in show.columns:
        cols += ["model_prediction", "prob_positive"]
    st.dataframe(show[cols], hide_index=True, height=420)


def page_predict():
    st.header("Live prediction")
    st.write("The model only uses the first " + str(config.MAX_LEN) + " tokens. Score 3 was not part of training.")

    if "live_text" not in st.session_state:
        st.session_state.live_text = ""

    st.write("Try an example")
    cols = st.columns(2)
    names = list(EXAMPLES)
    for i, name in enumerate(names):
        if cols[i % 2].button(name):
            st.session_state.live_text = EXAMPLES[name]
            st.rerun()

    text = st.text_area("Review text", key="live_text", height=140)
    if not text.strip():
        st.write("Type a review, or click an example.")
        return

    model = get_model()
    if model is None:
        st.warning("Missing models/lstm.keras. Run: python src/train.py")
        return

    with st.spinner("Scoring"):
        prob = float(predict_texts(model, [text])[0])
    if prob >= 0.5:
        label = "positive"
        conf = prob
    else:
        label = "negative"
        conf = 1 - prob
    st.write("Prediction:", label)
    st.write("Probability of positive:", round(prob, 3))
    st.write("Confidence in that label:", round(conf, 3))


def page_performance():
    st.header("Model performance")
    st.write("These tables are the files written by train.py and evaluate.py.")

    metrics = read_csv(os.path.join(config.REPORT_DIR, "metrics.csv"))
    if metrics is None:
        st.warning("Missing outputs/reports/metrics.csv. Run: python src/evaluate.py")
    else:
        st.subheader("Test metrics")
        st.dataframe(metrics, hide_index=True)

    comp = read_csv(os.path.join(config.REPORT_DIR, "model_comparison.csv"))
    if comp is None:
        st.warning("Missing outputs/reports/model_comparison.csv. Run: python src/train.py")
    else:
        st.subheader("LSTM vs TF-IDF logistic regression")
        st.dataframe(comp, hide_index=True)

    left, right = st.columns(2)
    with left:
        show_fig("confusion_matrix.png", "evaluate.py")
    with right:
        show_fig("roc_curve.png", "evaluate.py")
    show_fig("training_curves.png", "train.py")

    report_path = os.path.join(config.REPORT_DIR, "classification_report.txt")
    if os.path.exists(report_path):
        st.subheader("Classification report")
        with open(report_path, encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.warning("Missing outputs/reports/classification_report.txt. Run: python src/evaluate.py")


def page_errors():
    st.header("Error analysis")
    path = os.path.join(config.REPORT_DIR, "misclassified_reviews.csv")
    mistakes = read_csv(path)
    if mistakes is None:
        st.warning("Missing outputs/reports/misclassified_reviews.csv. Run: python src/evaluate.py")
    else:
        kind = st.selectbox("Which mistakes", ["All", "False positive", "False negative"])
        view = mistakes
        if kind == "False positive":
            view = mistakes[(mistakes["true label"] == 0) & (mistakes["predicted label"] == 1)]
        elif kind == "False negative":
            view = mistakes[(mistakes["true label"] == 1) & (mistakes["predicted label"] == 0)]
        st.write("False positive: model says positive, stars were 1 or 2.")
        st.write("False negative: model says negative, stars were 4 or 5.")
        if view.empty:
            st.write("None in this group.")
        else:
            page = page_slice(view, "error_page")
            st.dataframe(page, hide_index=True, height=420)

    for name, title in [
        ("performance_by_score.csv", "Accuracy by star rating"),
        ("performance_by_length.csv", "Accuracy by review length"),
        ("error_patterns.csv", "Keyword pattern counts"),
    ]:
        table = read_csv(os.path.join(config.REPORT_DIR, name))
        if table is not None:
            st.subheader(title)
            st.dataframe(table, hide_index=True)

    notes = os.path.join(config.REPORT_DIR, "label_limitations.txt")
    st.subheader("Limitations of the sentiment labels")
    if not os.path.exists(notes):
        st.warning("Missing outputs/reports/label_limitations.txt. Run: python src/evaluate.py")
        return
    with open(notes, encoding="utf-8") as f:
        text = f.read().strip()
    for para in text.split("\n\n"):
        st.write(para)


def main():
    st.set_page_config(page_title="E-commerce sentiment", layout="wide")
    st.sidebar.title("Reviews")
    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Review explorer",
            "Live prediction",
            "Model performance",
            "Error analysis",
        ],
    )
    if page == "Overview":
        page_overview()
    elif page == "Review explorer":
        page_explorer()
    elif page == "Live prediction":
        page_predict()
    elif page == "Model performance":
        page_performance()
    else:
        page_errors()


main()
