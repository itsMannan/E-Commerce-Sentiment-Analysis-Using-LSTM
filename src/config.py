import os

# Full file is about 500k reviews. 100k is enough to train on a laptop CPU.
# Raise this (up to 150000 or more) if you want a bigger run.
SAMPLE_SIZE = 100000

SEED = 42
MAX_WORDS = 20000
MAX_LEN = 200

# Fractions of the data after the neutral reviews are dropped.
# The test split is taken before balancing, so it keeps the real mix of scores.
TEST_SIZE = 0.15
VAL_SIZE = 0.15

BATCH_SIZE = 64
EPOCHS = 6
EMBED_DIM = 64
LSTM_UNITS = 64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "Reviews.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "models")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
REPORT_DIR = os.path.join(ROOT, "outputs", "reports")
