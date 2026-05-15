import re
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

TRAIN_PATH = "train_clean.csv"
TEST_PATH = "test_with_labels.csv"

def normalize_colname(col):
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())

def find_label_column(df):
    candidates = ["label", "target", "class", "category", "y", "sentiment"]
    normalized = {c: normalize_colname(c) for c in df.columns}
    for preferred in candidates:
        p = normalize_colname(preferred)
        for col, norm in normalized.items():
            if norm == p:
                return col
    for col, norm in normalized.items():
        if any(k in norm for k in ["label", "target", "class", "category", "sentiment"]):
            return col
    raise ValueError("Could not infer label column.")

def find_text_column(df):
    preferred = ["review_clean", "review", "text", "content", "message", "body", "comment"]
    normalized = {c: normalize_colname(c) for c in df.columns}
    for name in preferred:
        p = normalize_colname(name)
        for col, norm in normalized.items():
            if norm == p:
                return col
    for col, norm in normalized.items():
        if any(k in norm for k in ["review", "text", "content", "message", "comment"]):
            return col
    raise ValueError("Could not infer text column.")

train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
test_df = pd.read_csv(TEST_PATH, low_memory=False)

train_text_col = find_text_column(train_df)
train_label_col = find_label_column(train_df)
test_text_col = find_text_column(test_df)
test_label_col = find_label_column(test_df)

X_train = train_df[train_text_col].fillna("").astype(str)
y_train = train_df[train_label_col].astype(str)

X_test = test_df[test_text_col].fillna("").astype(str)
y_test = test_df[test_label_col].astype(str)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("clf", LinearSVC(class_weight="balanced"))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro F1:", round(f1_score(y_test, pred, average="macro"), 4))
print("Weighted F1:", round(f1_score(y_test, pred, average="weighted"), 4))
print()
print(classification_report(y_test, pred, digits=4))
