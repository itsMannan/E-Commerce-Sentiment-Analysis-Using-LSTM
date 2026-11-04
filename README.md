# Deep Learning-Based E-Commerce Sentiment Analysis for Customer Insights

Online shops get more reviews than anyone can read. I wanted a practical way to sort those reviews into positive and negative, then see what people actually complain about. This project trains a small LSTM on Amazon Fine Food reviews and compares it with a simpler TF-IDF logistic regression, so I can tell whether the neural net is earning its keep.

Stars are the label. That is convenient and noisy. A 1-star review is sometimes about a cracked jar, not the taste.

## Dataset

I used the Amazon Fine Food Reviews dataset from Kaggle (`snap/amazon-fine-food-reviews`).

The raw file is `Reviews.csv`. Put it here:

```text
data/raw/Reviews.csv
```

The code does not download it. Columns I use: `ProductId`, `UserId`, `ProfileName`, `Score`, `Time`, `Summary`, `Text`.

There is no product category column. For the category charts I tag reviews with keywords (coffee, tea, chocolate, snacks, pet food, baby food, sauces, pasta, other). The first match wins, and everything else is "other". "hot dog" is rewritten before the pet rule so it is not counted as pet food. These labels are approximate. A review that mentions coffee and chocolate only gets one tag.

## Project structure

```text
data/raw/            Reviews.csv (not in git)
data/processed/      cleaned reviews, train, val, test, neutral
models/              LSTM, vocabulary, TF-IDF, logistic regression
outputs/figures/     charts
outputs/reports/     metrics, mistakes, limitations note
src/config.py        sample size, sequence length, paths, seed
src/data_prep.py
src/eda.py
src/train.py
src/evaluate.py
src/insights.py
app.py               Streamlit dashboard
```

## Setup

Python 3.11 or newer. I used 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the scripts from the project folder, in this order:

```bash
python src/data_prep.py
python src/eda.py
python src/train.py
python src/evaluate.py
python src/insights.py
streamlit run app.py
```

`data_prep.py` can take a few minutes. `train.py` is the slow one (often half an hour to a couple of hours on a laptop CPU). The Keras bar shows progress.

If training is too slow, lower `EPOCHS` in `src/config.py` and run `train.py` again. `SAMPLE_SIZE` caps the balanced training set (currently 100000). Change it, then rerun `data_prep.py` before training. The validation and test sets are not capped by that number. They stay at the real mix of stars.

## Methodology

I drop duplicate rows that share the same user, profile name, time, and text. I drop rows with no review text.

Labels:

- score 1 or 2: negative (0)
- score 4 or 5: positive (1)
- score 3: left out of the model and saved in `data/processed/neutral.csv`

I join `Summary` and `Text` with `" . "` because the summary is often the short opinion and the body is the detail. Cleaning is lowercase, HTML tags, URLs, and extra spaces. I do not strip stopwords. "not good" would become "good".

I split off validation and test first, stratified, so they keep the real positive-heavy mix. I only downsample the training set, so the two classes match, and I cap that balanced set with `SAMPLE_SIZE`. Test accuracy is the honest number. It can look high because most reviews are 5 stars.

The LSTM is small on purpose: embedding (64), one LSTM layer (64), dropout, a dense layer of 32 with relu, dropout, then a sigmoid. Vocabulary cap is 20,000. Reviews are cut at 200 tokens. Padding is index 0 and the embedding masks it, so the LSTM skips pad steps. I train with binary crossentropy and Adam, and I stop early if validation loss stalls. The best epoch is the saved model.

The baseline is TF-IDF unigrams and bigrams (20,000 features) plus logistic regression. Bigrams give the linear model a chance at "not good". Both models see the same test file. The scores land in `outputs/reports/model_comparison.csv`. If the baseline wins or ties, that is the result.

Random seed is 42 for Python, NumPy, and TensorFlow.

## Results

I am not writing scores in here before the scripts have run. After `train.py` and `evaluate.py`, copy the numbers from the csv files.

From `outputs/reports/metrics.csv` (LSTM on the held-out test set):

- accuracy: `<fill in from outputs/reports/metrics.csv>`
- precision (positive class): `<fill in from outputs/reports/metrics.csv>`
- recall (positive class): `<fill in from outputs/reports/metrics.csv>`
- F1: `<fill in from outputs/reports/metrics.csv>`
- ROC-AUC: `<fill in from outputs/reports/metrics.csv>`

From `outputs/reports/model_comparison.csv`:

- LSTM accuracy / F1 / ROC-AUC: `<fill in from outputs/reports/model_comparison.csv>`
- TF-IDF logistic regression accuracy / F1 / ROC-AUC: `<fill in from outputs/reports/model_comparison.csv>`

Training curves: `outputs/figures/training_curves.png`. History: `outputs/reports/train_history.csv`.

## Error analysis and label limits

`evaluate.py` writes the mistake list, accuracy by star and by length, and a short limitations note. I fill this section in only after that run.

- Misclassified reviews: `outputs/reports/misclassified_reviews.csv`
- Accuracy by star: `outputs/reports/performance_by_score.csv`
- Accuracy by length: `outputs/reports/performance_by_length.csv`
- Keyword pattern counts: `outputs/reports/error_patterns.csv`
- Notes: `outputs/reports/label_limitations.txt`

Paste from `label_limitations.txt` after your run:

`<paste outputs/reports/label_limitations.txt here>`

What I already know from the way the labels are built, before looking at scores: score 3 is dropped because it is ambiguous, the raw data is heavy on 5-star reviews, and a low star can be about shipping rather than the food. The text file is where I check those claims against this run. The pattern counts are keyword matches, not a full read of every mistake.

Word and theme charts come from `insights.py` (`outputs/figures/`). Stopwords are removed only on those word-count charts, and I put "not", "no", "never", and "nor" back so negation can still show up. The logistic regression chart uses the real coefficients, stopwords included.

## Dashboard

`streamlit run app.py`

Pages: overview, review explorer, live prediction, model performance, error analysis. If a file is missing, the page says which script to run. Overview numbers are read from the cleaned data and the report csvs. The explorer scores one page of reviews at a time.

## Possible next steps

- Keep score 3 and train a 3-class model.
- Start from pretrained embeddings such as GloVe instead of learning them from scratch.
- Split a review into aspects (taste, price, shipping) instead of one label for the whole text.
- Do something more serious about sarcasm and mixed reviews than a keyword list.

## CV line

Title: Deep Learning-Based E-Commerce Sentiment Analysis for Customer Insights

Bullet: Built an LSTM-based sentiment classification pipeline for customer reviews to analyze feedback patterns and explore applications in product evaluation and customer experience management.
