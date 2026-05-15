import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.base import clone

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# =========================================================
# Configuration
# =========================================================
TRAIN_PATH = "train_clean.csv"
TEST_PATH = "test_with_labels.csv"
OUTPUT_DIR = "outputs"

RANDOM_STATE = 42
N_SPLITS = 5
TOP_N_WORDS = 20
TOP_N_NGRAMS = 15
TOP_N_FEATURES = 25
MISCLASSIFIED_PER_CLASS = 10
MAX_TEXT_EXAMPLES = 3


# =========================================================
# Utility helpers
# =========================================================
def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120]


def normalize_colname(col):
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())


def find_label_column(df):
    candidates = ["label", "target", "class", "category", "y", "sentiment"]
    normalized = {c: normalize_colname(c) for c in df.columns}

    for preferred in candidates:
        preferred_norm = normalize_colname(preferred)
        for col, norm in normalized.items():
            if norm == preferred_norm:
                return col

    for col, norm in normalized.items():
        if any(key in norm for key in ["label", "target", "class", "category", "sentiment"]):
            return col

    raise ValueError("Could not infer label column.")


def choose_text_column(df, dataset_name="dataset"):
    preferred_order = [
        "review_clean",
        "review_clean_v2",
        "clean_review",
        "text_clean",
        "review",
        "text",
        "content",
        "message",
        "document",
        "body",
        "comment"
    ]

    normalized = {c: normalize_colname(c) for c in df.columns}

    for preferred in preferred_order:
        preferred_norm = normalize_colname(preferred)
        for col, norm in normalized.items():
            if norm == preferred_norm:
                series = df[col].fillna("").astype(str).str.strip()
                non_empty_ratio = series.ne("").mean()

                if preferred_norm in {"reviewclean", "reviewcleanv2", "cleanreview", "textclean"}:
                    if non_empty_ratio > 0.50:
                        print(f"[{dataset_name}] Using cleaned text column: {col}")
                        return col
                else:
                    if non_empty_ratio > 0.50:
                        print(f"[{dataset_name}] Using text column: {col}")
                        return col

    for col in df.columns:
        norm = normalize_colname(col)
        if any(key in norm for key in ["reviewclean", "textclean", "cleanreview"]):
            series = df[col].fillna("").astype(str).str.strip()
            if series.ne("").mean() > 0.50:
                print(f"[{dataset_name}] Using fallback cleaned text column: {col}")
                return col

    for col in df.columns:
        norm = normalize_colname(col)
        if any(key in norm for key in ["review", "text", "content", "message", "comment"]):
            series = df[col].fillna("").astype(str).str.strip()
            if series.ne("").mean() > 0.50:
                print(f"[{dataset_name}] Using fallback raw text column: {col}")
                return col

    raise ValueError(f"Could not infer text column for {dataset_name}.")


def basic_clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_simple(text):
    text = basic_clean_text(text).lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


def get_text_stats(series):
    s = series.fillna("").astype(str)
    return pd.DataFrame({
        "char_count": s.str.len(),
        "word_count": s.str.split().apply(len)
    })


def quick_text_quality_check(df, text_col, dataset_name, top_n=20):
    print_header(f"TEXT COLUMN CHECK: {dataset_name}")

    series = df[text_col].fillna("").astype(str)
    empty_count = series.str.strip().eq("").sum()
    print(f"Chosen text column: {text_col}")
    print(f"Empty rows: {empty_count} / {len(df)}")

    tokens = series.str.split().explode()
    tokens = tokens[tokens.notna()]
    top_words = tokens.value_counts().head(top_n)

    print("\nTop words from chosen text column:")
    print(top_words)

    suspicious = ["the", "and", "i", "a", "to", "was", "it", "of", "is", "for"]
    suspicious_counts = {w: int((tokens == w).sum()) for w in suspicious}
    print("\nSuspicious stopword counts:")
    print(suspicious_counts)

    if sum(suspicious_counts.values()) > 0:
        print("WARNING: Chosen text column still contains common stopwords.")
        print("If this is train_clean.csv, verify that EDA is using the cleaned column and not raw review text.")


def warn_class_imbalance(y, dataset_name):
    counts = y.value_counts()
    ratios = counts / counts.sum()
    min_ratio = ratios.min()
    imbalance_ratio = counts.max() / max(counts.min(), 1)

    print(f"\nClass distribution for {dataset_name}:")
    print(pd.DataFrame({
        "count": counts,
        "pct": (ratios * 100).round(2)
    }))

    if min_ratio < 0.10 or imbalance_ratio >= 3:
        print(f"WARNING: Potential class imbalance detected in {dataset_name}.")
        print(f"Smallest class %: {min_ratio:.3f}, imbalance ratio: {imbalance_ratio:.2f}")
    else:
        print(f"No severe class imbalance warning for {dataset_name}.")


def dataset_validation(df, dataset_name, text_col, label_col):
    print_header(f"DATA VALIDATION: {dataset_name}")

    print(f"Shape: {df.shape}")
    print(f"Text column: {text_col}")
    print(f"Label column: {label_col}")

    print("\nMissing values by column:")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0] if (missing > 0).any() else "No missing values.")

    duplicate_rows = df.duplicated().sum()
    duplicate_text_label = df.duplicated(subset=[text_col, label_col]).sum()
    empty_text_rows = df[text_col].fillna("").astype(str).str.strip().eq("").sum()

    print(f"\nDuplicate full rows: {duplicate_rows}")
    print(f"Duplicate text+label rows: {duplicate_text_label}")
    print(f"Empty text rows: {empty_text_rows}")

    warn_class_imbalance(df[label_col], dataset_name)


# =========================================================
# EDA
# =========================================================
def plot_length_distributions(df, text_col, label_col, dataset_name, output_dir):
    stats_df = get_text_stats(df[text_col])
    temp = df[[label_col]].copy()
    temp["char_count"] = stats_df["char_count"]
    temp["word_count"] = stats_df["word_count"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.boxplot(data=temp, x=label_col, y="char_count", ax=axes[0])
    axes[0].set_title(f"{dataset_name}: Character Count by Class")
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(data=temp, x=label_col, y="word_count", ax=axes[1])
    axes[1].set_title(f"{dataset_name}: Word Count by Class")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{safe_filename(dataset_name)}_length_by_class.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    summary = temp.groupby(label_col)[["char_count", "word_count"]].agg(["mean", "median", "std", "min", "max"])
    print("\nLength summary by class:")
    print(summary.round(2))
    return summary


def top_terms(series, top_n=20):
    counter = Counter()
    for text in series.fillna("").astype(str):
        counter.update(tokenize_simple(text))
    return pd.DataFrame(counter.most_common(top_n), columns=["term", "count"])


def top_terms_per_class(df, text_col, label_col, top_n=20):
    out = []
    for label in sorted(df[label_col].astype(str).unique()):
        temp = top_terms(df[df[label_col].astype(str) == str(label)][text_col], top_n=top_n)
        temp[label_col] = label
        out.append(temp)
    return pd.concat(out, ignore_index=True)


def top_ngram_df(df, text_col, label_col, ngram_range=(2, 2), top_n=15):
    records = []
    labels = sorted(df[label_col].astype(str).unique())

    for label in labels:
        subset = df[df[label_col].astype(str) == str(label)][text_col].fillna("").astype(str)
        if subset.shape[0] == 0:
            continue

        vec = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )

        try:
            X = vec.fit_transform(subset)
            scores = np.asarray(X.sum(axis=0)).ravel()
            feats = np.array(vec.get_feature_names_out())
            top_idx = np.argsort(scores)[::-1][:top_n]

            for idx in top_idx:
                records.append({
                    "class": label,
                    "ngram": feats[idx],
                    "score": float(scores[idx]),
                    "ngram_type": f"{ngram_range[0]}-gram"
                })
        except Exception:
            continue

    return pd.DataFrame(records)


def show_shortest_longest_examples(df, text_col, label_col):
    temp = df[[text_col, label_col]].copy()
    temp["text"] = temp[text_col].fillna("").astype(str)
    temp["char_count"] = temp["text"].str.len()

    shortest = temp.sort_values("char_count", ascending=True).head(MAX_TEXT_EXAMPLES)
    longest = temp.sort_values("char_count", ascending=False).head(MAX_TEXT_EXAMPLES)

    print("\nShortest text examples:")
    for _, row in shortest.iterrows():
        print(f"- Label={row[label_col]} | chars={row['char_count']} | text={row['text'][:300]}")

    print("\nLongest text examples:")
    for _, row in longest.iterrows():
        print(f"- Label={row[label_col]} | chars={row['char_count']} | text={row['text'][:300]}...")


def summarize_eda_findings(df, text_col, label_col):
    stats_df = get_text_stats(df[text_col])
    temp = df[[label_col]].copy()
    temp["char_count"] = stats_df["char_count"]
    temp["word_count"] = stats_df["word_count"]

    by_class = temp.groupby(label_col)[["char_count", "word_count"]].mean().round(1)
    vocab_sizes = {}

    for label in sorted(df[label_col].astype(str).unique()):
        tokens = []
        subset = df[df[label_col].astype(str) == str(label)][text_col].fillna("").astype(str)
        for text in subset:
            tokens.extend(tokenize_simple(text))
        vocab_sizes[str(label)] = len(set(tokens))

    print_header("EDA FINDINGS")
    print("Plain-English findings:")
    print("- Check whether some classes have much shorter reviews; those classes often get lower recall.")
    print("- If classes use overlapping restaurant vocabulary, confusion will rise unless n-grams capture more context.")
    print("- Strong differences in average length can indicate the model is using style/verbosity as a shortcut.")
    print("- Small classes with smaller vocabularies often need class weights or richer n-grams.")
    print("\nAverage length by class:")
    print(by_class)
    print("\nApproximate vocabulary size by class:")
    print(pd.Series(vocab_sizes).sort_values(ascending=False))


def run_text_eda(df, text_col, label_col, dataset_name, output_dir):
    print_header(f"TEXT EDA: {dataset_name}")
    quick_text_quality_check(df, text_col, dataset_name, top_n=TOP_N_WORDS)

    length_summary = plot_length_distributions(df, text_col, label_col, dataset_name, output_dir)

    overall_words = top_terms(df[text_col], top_n=TOP_N_WORDS)
    overall_words.to_csv(os.path.join(output_dir, f"{safe_filename(dataset_name)}_top_words_overall.csv"), index=False)
    print("\nTop words overall:")
    print(overall_words)

    per_class_words = top_terms_per_class(df, text_col, label_col, top_n=TOP_N_WORDS)
    per_class_words.to_csv(os.path.join(output_dir, f"{safe_filename(dataset_name)}_top_words_per_class.csv"), index=False)
    print("\nTop words per class:")
    print(per_class_words.head(60))

    bigrams = top_ngram_df(df, text_col, label_col, ngram_range=(2, 2), top_n=TOP_N_NGRAMS)
    trigrams = top_ngram_df(df, text_col, label_col, ngram_range=(3, 3), top_n=TOP_N_NGRAMS)

    if not bigrams.empty:
        bigrams.to_csv(os.path.join(output_dir, f"{safe_filename(dataset_name)}_top_bigrams_per_class.csv"), index=False)
        print("\nTop bigrams per class:")
        print(bigrams.head(60))

    if not trigrams.empty:
        trigrams.to_csv(os.path.join(output_dir, f"{safe_filename(dataset_name)}_top_trigrams_per_class.csv"), index=False)
        print("\nTop trigrams per class:")
        print(trigrams.head(60))

    show_shortest_longest_examples(df, text_col, label_col)
    summarize_eda_findings(df, text_col, label_col)

    return {
        "length_summary": length_summary,
        "overall_words": overall_words,
        "per_class_words": per_class_words,
        "bigrams": bigrams,
        "trigrams": trigrams
    }


# =========================================================
# Modeling
# =========================================================
def build_model_candidates():
    models = {}

    models["tfidf_word_lr"] = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    models["tfidf_word_svc"] = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(
            class_weight="balanced"
        ))
    ])

    models["tfidf_wordchar_lr"] = Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words=None,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True
            ))
        ])),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    models["tfidf_wordchar_svc"] = Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words=None,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True
            ))
        ])),
        ("clf", LinearSVC(
            class_weight="balanced"
        ))
    ])

    return models


def compare_models(X_train, y_train, output_dir):
    print_header("BASELINE MODEL COMPARISON")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy"
    }

    models = build_model_candidates()
    results = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=None,
            return_train_score=False
        )

        results.append({
            "model": name,
            "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
            "cv_macro_f1_std": scores["test_macro_f1"].std(),
            "cv_weighted_f1_mean": scores["test_weighted_f1"].mean(),
            "cv_weighted_f1_std": scores["test_weighted_f1"].std(),
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std()
        })

    results_df = pd.DataFrame(results).sort_values(
        by=["cv_macro_f1_mean", "cv_weighted_f1_mean"],
        ascending=False
    ).reset_index(drop=True)

    print(results_df.round(4))
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]

    print(f"\nSelected best baseline: {best_name}")
    return best_name, best_model, results_df


def fit_best_model(model, X_train, y_train):
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    return fitted


# =========================================================
# Feature importance
# =========================================================
def get_feature_names_from_pipeline(model):
    if "tfidf" in model.named_steps:
        return model.named_steps["tfidf"].get_feature_names_out()

    if "features" in model.named_steps:
        feature_union = model.named_steps["features"]
        names = []
        for transformer_name, transformer in feature_union.transformer_list:
            transformer_features = transformer.get_feature_names_out()
            transformer_features = [f"{transformer_name}__{f}" for f in transformer_features]
            names.extend(transformer_features)
        return np.array(names)

    raise ValueError("Could not extract feature names from model.")


def extract_feature_importance(model, class_labels, top_n=25):
    print_header("FEATURE IMPORTANCE")

    clf = model.named_steps["clf"]
    feature_names = get_feature_names_from_pipeline(model)

    if not hasattr(clf, "coef_"):
        print("Model does not expose coefficients. Skipping feature importance.")
        return pd.DataFrame()

    coef = clf.coef_
    rows = []

    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    if len(class_labels) == 2 and coef.shape[0] == 1:
        positive_class = class_labels[1]
        negative_class = class_labels[0]

        top_pos_idx = np.argsort(coef[0])[::-1][:top_n]
        top_neg_idx = np.argsort(coef[0])[:top_n]

        for idx in top_pos_idx:
            rows.append({
                "class": positive_class,
                "feature": feature_names[idx],
                "coefficient": float(coef[0, idx]),
                "direction": "positive_for_class"
            })
        for idx in top_neg_idx:
            rows.append({
                "class": negative_class,
                "feature": feature_names[idx],
                "coefficient": float(coef[0, idx]),
                "direction": "positive_for_class"
            })
    else:
        for class_idx, class_name in enumerate(class_labels):
            class_coef = coef[class_idx]
            top_idx = np.argsort(class_coef)[::-1][:top_n]
            for idx in top_idx:
                rows.append({
                    "class": class_name,
                    "feature": feature_names[idx],
                    "coefficient": float(class_coef[idx]),
                    "direction": "positive_for_class"
                })

    importance_df = pd.DataFrame(rows).sort_values(["class", "coefficient"], ascending=[True, False])
    print(importance_df.head(100))
    return importance_df


# =========================================================
# Evaluation and error analysis
# =========================================================
def get_prediction_confidence(model, X):
    clf = model.named_steps["clf"]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        conf = probs.max(axis=1)
        return conf, probs

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        if np.ndim(scores) == 1:
            return np.abs(scores), None
        return np.max(scores, axis=1), None

    return None, None


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return cm_df


def most_confused_pairs(cm_df):
    rows = []
    labels = list(cm_df.index)

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = int(cm_df.iloc[i, j])
            if count > 0:
                rows.append({
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count
                })

    if not rows:
        return pd.DataFrame(columns=["true_label", "pred_label", "count"])

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def collect_misclassified_examples(df_eval, text_col, label_col, pred_col, conf_col=None, per_class=10):
    rows = []
    labels = sorted(df_eval[label_col].astype(str).unique())

    for current_class in labels:
        subset = df_eval[
            (df_eval[label_col].astype(str) == str(current_class)) &
            (df_eval[pred_col].astype(str) != str(current_class))
        ].copy()

        subset["text_len"] = subset[text_col].fillna("").astype(str).str.len()
        subset = subset.sort_values("text_len", ascending=True)

        for _, row in subset.head(per_class).iterrows():
            rows.append({
                "focus_class": current_class,
                "true_label": row[label_col],
                "pred_label": row[pred_col],
                "confidence": row[conf_col] if conf_col and conf_col in row.index else np.nan,
                "text_len": len(str(row[text_col])),
                "text": str(row[text_col])[:1500]
            })

    return pd.DataFrame(rows)


def false_positive_false_negative_tables(df_eval, label_col, pred_col):
    labels = sorted(df_eval[label_col].astype(str).unique())
    rows = []

    for cls in labels:
        true_cls = df_eval[label_col].astype(str) == str(cls)
        pred_cls = df_eval[pred_col].astype(str) == str(cls)

        fp = ((~true_cls) & pred_cls).sum()
        fn = (true_cls & (~pred_cls)).sum()
        tp = (true_cls & pred_cls).sum()

        rows.append({
            "class": cls,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        })

    return pd.DataFrame(rows).sort_values("class")


def analyze_error_patterns(df_eval, text_col, label_col, pred_col):
    print_header("ERROR PATTERN ANALYSIS")

    df_eval = df_eval.copy()
    df_eval["is_error"] = df_eval[label_col].astype(str) != df_eval[pred_col].astype(str)
    df_eval["char_count"] = df_eval[text_col].fillna("").astype(str).str.len()
    df_eval["word_count"] = df_eval[text_col].fillna("").astype(str).str.split().apply(len)

    error_rate = df_eval["is_error"].mean()
    print(f"Overall error rate: {error_rate:.4f}")

    length_summary = df_eval.groupby("is_error")[["char_count", "word_count"]].mean().round(2)
    print("\nAverage text length for correct vs incorrect predictions:")
    print(length_summary)

    print("\nLikely patterns to inspect:")
    print("- Short texts: if error texts are much shorter, recall may suffer on sparse examples.")
    print("- Overlapping vocabulary: heavy confusion between related cuisines/categories often means unigram features are not enough.")
    print("- Class imbalance: high false negatives on rare classes often indicate recall problems.")
    print("- Preprocessing loss: aggressive cleaning can remove negation, domain markers, or short informative tokens.")
    print("- Rare token issues: unusual spellings, names, slang, and misspellings often benefit from char n-grams.")
    print("- Label noise: repeated contradictory examples or odd outliers may indicate mislabeled samples.")

    return length_summary


def evaluate_on_test(model, X_test, y_test, raw_test_df, text_col, output_dir):
    print_header("FINAL TEST EVALUATION")

    y_pred = model.predict(X_test)
    confidence, _ = get_prediction_confidence(model, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")

    labels = sorted(pd.Series(y_test).astype(str).unique())
    p, r, f1_vals, s = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)

    metrics_df = pd.DataFrame({
        "class": labels,
        "precision": p,
        "recall": r,
        "f1": f1_vals,
        "support": s
    }).sort_values("class")

    print("\nPer-class metrics:")
    print(metrics_df.round(4))
    metrics_df.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    cm_df = plot_confusion_matrix(y_test, y_pred, labels, cm_path)

    confused_pairs_df = most_confused_pairs(cm_df)
    confused_pairs_df.to_csv(os.path.join(output_dir, "most_confused_class_pairs.csv"), index=False)

    print("\nMost confused class pairs:")
    print(confused_pairs_df.head(20))

    eval_df = raw_test_df.copy()
    eval_df["true_label"] = y_test.values if isinstance(y_test, pd.Series) else y_test
    eval_df["pred_label"] = y_pred
    if confidence is not None:
        eval_df["confidence"] = confidence

    fp_fn_df = false_positive_false_negative_tables(eval_df, "true_label", "pred_label")
    fp_fn_df.to_csv(os.path.join(output_dir, "false_positive_false_negative_summary.csv"), index=False)

    print("\nFalse positives / false negatives by class:")
    print(fp_fn_df)

    misclassified_df = eval_df[eval_df["true_label"].astype(str) != eval_df["pred_label"].astype(str)].copy()
    misclassified_df.to_csv(os.path.join(output_dir, "all_misclassified_examples.csv"), index=False)

    sample_misclassified_df = collect_misclassified_examples(
        eval_df,
        text_col=text_col,
        label_col="true_label",
        pred_col="pred_label",
        conf_col="confidence" if "confidence" in eval_df.columns else None,
        per_class=MISCLASSIFIED_PER_CLASS
    )
    sample_misclassified_df.to_csv(os.path.join(output_dir, "sample_misclassified_examples_by_class.csv"), index=False)

    print("\nSample misclassified examples by class:")
    print(sample_misclassified_df.head(50))

    analyze_error_patterns(eval_df, text_col, "true_label", "pred_label")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_metrics": metrics_df,
        "confusion_matrix": cm_df,
        "confused_pairs": confused_pairs_df,
        "fp_fn": fp_fn_df,
        "misclassified_examples": sample_misclassified_df
    }


# =========================================================
# Recommendations
# =========================================================
def print_recommended_next_steps(train_df, text_col, label_col, model_name):
    print_header("RECOMMENDED NEXT STEPS TO IMPROVE F1")

    class_counts = train_df[label_col].value_counts()
    imbalance_ratio = class_counts.max() / max(class_counts.min(), 1)

    print("1. Keep interpretable TF-IDF baselines as your main diagnostic tool before moving to embeddings.")
    print("   They expose top terms, confusion patterns, and per-class recall failures very clearly.")

    print("\n2. Revisit preprocessing.")
    print("   Avoid overly aggressive stopword removal and dropping all short tokens.")
    print("   In reviews, short tokens and function words can still carry signal, and aggressive cleaning may remove negation or domain cues.")

    print("\n3. Use class weights and optimize for macro F1.")
    print("   This matters most when minority classes are missing recall.")
    if imbalance_ratio >= 3:
        print(f"   Your training data likely has meaningful imbalance (ratio about {imbalance_ratio:.2f}).")

    print("\n4. Expand word n-grams and keep char n-grams in the search space.")
    print("   Word bigrams help with phrases; char n-grams help with misspellings, cuisine names, slang, and noisy text.")

    print("\n5. Inspect the most confused class pairs.")
    print("   If classes are semantically close, add phrase-level features or review whether labels are too overlapping.")

    print("\n6. Inspect false negatives per class first.")
    print("   For F1 improvement, recall failures in underperforming classes are often the highest-value place to work.")

    print("\n7. Remove duplicates and review suspicious labels.")
    print("   Duplicate text with conflicting labels and obviously inconsistent samples can cap F1 even with better models.")

    print("\n8. If the task is binary, tune the decision threshold after baseline selection.")
    print("   Threshold tuning can improve F1 materially when precision-recall tradeoffs are uneven.")

    print("\n9. Only after these simple baselines are exhausted, try stronger models.")
    print("   Examples: calibrated linear models, SGD variants, or transformer models if dataset size and latency allow.")

    print("\nWhy averaged Word2Vec may underperform TF-IDF on this kind of task:")
    print("- Averaging embeddings discards which specific words were present and how often they mattered.")
    print("- It weakens rare but highly predictive tokens, which are often exactly what helps class-level F1.")
    print("- It removes much of the phrase-level signal unless the embeddings are very strong and the data is large.")
    print("- TF-IDF with linear models is often stronger on medium-sized tabular text tasks because it preserves sparse, highly discriminative lexical cues.")


# =========================================================
# Main
# =========================================================
def main():
    ensure_output_dir(OUTPUT_DIR)

    print_header("LOADING DATA")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    train_label_col = find_label_column(train_df)
    test_label_col = find_label_column(test_df)

    train_text_col = choose_text_column(train_df, "train")
    test_text_col = choose_text_column(test_df, "test")

    train_df[train_text_col] = train_df[train_text_col].apply(basic_clean_text)
    test_df[test_text_col] = test_df[test_text_col].apply(basic_clean_text)

    dataset_validation(train_df, "train.csv", train_text_col, train_label_col)
    dataset_validation(test_df, "test_with_labels.csv", test_text_col, test_label_col)

    run_text_eda(train_df, train_text_col, train_label_col, "train.csv", OUTPUT_DIR)
    run_text_eda(test_df, test_text_col, test_label_col, "test_with_labels.csv", OUTPUT_DIR)

    X_train = train_df[train_text_col].fillna("").astype(str)
    y_train = train_df[train_label_col].astype(str)

    X_test = test_df[test_text_col].fillna("").astype(str)
    y_test = test_df[test_label_col].astype(str)

    best_name, best_model, comparison_df = compare_models(X_train, y_train, OUTPUT_DIR)
    best_fitted = fit_best_model(best_model, X_train, y_train)

    feature_importance_df = extract_feature_importance(
        best_fitted,
        class_labels=np.sort(y_train.unique()),
        top_n=TOP_N_FEATURES
    )
    if not feature_importance_df.empty:
        feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_top_terms.csv"), index=False)

    evaluate_on_test(
        best_fitted,
        X_test,
        y_test,
        raw_test_df=test_df,
        text_col=test_text_col,
        output_dir=OUTPUT_DIR
    )

    print_recommended_next_steps(train_df, train_text_col, train_label_col, best_name)

    print_header("FILES SAVED")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print("- model_comparison.csv")
    print("- per_class_metrics.csv")
    print("- confusion_matrix.png")
    print("- most_confused_class_pairs.csv")
    print("- false_positive_false_negative_summary.csv")
    print("- all_misclassified_examples.csv")
    print("- sample_misclassified_examples_by_class.csv")
    print("- feature_importance_top_terms.csv")
    print("- train.csv_top_words_overall.csv / per-class / n-grams")
    print("- test_with_labels.csv_top_words_overall.csv / per-class / n-grams")


if __name__ == "__main__":
    main()
