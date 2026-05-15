import pandas as pd
import numpy as np
import ast
import re
import os
import html
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE = "train.csv"
OUTPUT_FILE = "train_clean.csv"

DICT_COLUMNS = {
    "attributes.Ambience": "ambience",
    "attributes.BusinessParking": "parking",
    "attributes.GoodForMeal": "goodformeal",
    "attributes.Music": "music",
    "attributes.BestNights": "bestnights",
    "attributes.DietaryRestrictions": "dietary",
    "hours": "hours",
}

DROP_COLS = ["attributes"]

HOURS_COLS = [
    "hours.Monday", "hours.Tuesday", "hours.Wednesday",
    "hours.Thursday", "hours.Friday", "hours.Saturday", "hours.Sunday"
]

TRUE_VALS = {"true", "1", "yes"}
FALSE_VALS = {"false", "0", "no"}

NEGATION_WORDS = {"no", "not", "nor", "never"}

# ─────────────────────────────────────────────────────────────────────────────

def clean_value(val):
    if not isinstance(val, str):
        return val
    val = val.strip()

    patterns = [
        r'b""(.*?)""',
        r"b'(.*?)'",
        r'b"(.*?)"'
    ]
    for pattern in patterns:
        m = re.fullmatch(pattern, val, re.DOTALL)
        if m:
            return m.group(1)
    return val


def normalize_ustr(val):
    if isinstance(val, str):
        m = re.fullmatch(r"u'(.*)'", val, re.DOTALL)
        if m:
            return m.group(1)
    return val


def strip_bare_quotes(val):
    if isinstance(val, str):
        m = re.fullmatch(r"'(.*)'", val, re.DOTALL)
        if m:
            return m.group(1)
    return val


def try_parse_dict(val):
    if not isinstance(val, str):
        return None
    val = val.strip()
    if not (val.startswith("{") and val.endswith("}")):
        return None

    normalized = re.sub(r"\bu'([^']*)'", r"'\1'", val)
    try:
        result = ast.literal_eval(normalized)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


def expand_dict_column(df, col, prefix):
    if col not in df.columns:
        return df

    parsed = df[col].apply(try_parse_dict)
    expanded = parsed.apply(lambda d: d if isinstance(d, dict) else {})
    expanded_df = pd.json_normalize(expanded).add_prefix(f"{prefix}.")
    expanded_df.index = df.index

    return pd.concat([df.drop(columns=[col]), expanded_df], axis=1)


def dedup_columns(df):
    seen = set()
    keep = []

    for i, col in enumerate(df.columns):
        if col not in seen:
            seen.add(col)
            keep.append(i)

    dupes = len(df.columns) - len(keep)
    if dupes:
        print(f"  Dropping {dupes} duplicate column(s).")

    return df.iloc[:, keep]


def is_bool_col(series):
    if not isinstance(series, pd.Series):
        return False
    if series.dtype != object:
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    unique_vals = set(non_null.astype(str).str.strip().str.lower().unique())
    return unique_vals.issubset(TRUE_VALS | FALSE_VALS)


def to_bool_int(series):
    mapping = {**{v: 1 for v in TRUE_VALS}, **{v: 0 for v in FALSE_VALS}}
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0)
        .astype(int)
    )


def parse_time_to_minutes(t: str) -> int:
    h, m = t.strip().split(":")
    return int(h) * 60 + int(m)


def hours_open(time_range) -> float:
    if not isinstance(time_range, str) or "-" not in time_range:
        return np.nan

    m = re.fullmatch(r"(\d+:\d+)-(\d+:\d+)", time_range.strip())
    if not m:
        return np.nan

    open_min = parse_time_to_minutes(m.group(1))
    close_min = parse_time_to_minutes(m.group(2))

    if open_min == 0 and close_min == 0:
        return 0.0
    if close_min <= open_min:
        close_min += 24 * 60

    return round((close_min - open_min) / 60, 4)


def build_stop_words() -> set:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    sw = set(stopwords.words("english"))

    # Keep negation words because they often carry meaning in reviews.
    sw = sw - NEGATION_WORDS

    # Add light, safe extras only.
    extra = {
        "im", "id", "ive", "ill", "youre", "youve", "youll",
        "theyre", "theyve", "weve", "thats", "theres", "heres",
        "cant", "wont", "didnt", "doesnt", "isnt", "wasnt",
        "couldnt", "wouldnt", "shouldnt", "nt", "00"
    }
    sw.update(extra)

    return sw


def fix_common_encoding_artifacts(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    replacements = {
        "xc3 xa9": "e",
        "xc3 xa8": "e",
        "xc3 xa0": "a",
        "xc3 xa1": "a",
        "xc3 xa2": "a",
        "xc3 xa4": "a",
        "xc3 xb1": "n",
        "xc3 xb3": "o",
        "xc3 xb6": "o",
        "xc3 xb8": "o",
        "xc3 xbc": "u",
        "xc3 xa7": "c",
        "xc2 xa0": " ",
        "\\xc3\\xa9": "e",
        "\\xc3\\xa8": "e",
        "\\xc3\\xa0": "a",
        "\\xc3\\xb1": "n",
        "\\xc2\\xa0": " ",
    }

    text = text.lower()
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove isolated byte-fragment tokens that survive replacement.
    text = re.sub(r"\bxc[0-9a-f]{1,2}\b", " ", text)
    text = re.sub(r"\bx[0-9a-f]{2}\b", " ", text)

    return text


def normalize_review_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if text == "" or text.lower() == "unknown":
        return ""

    text = clean_value(text)
    text = normalize_ustr(text)
    text = strip_bare_quotes(text)

    text = fix_common_encoding_artifacts(text)

    # Normalize quotes/apostrophes
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("`", "'")
            .replace("“", '"')
            .replace("”", '"')
    )

    # Normalize escaped and real whitespace
    text = (
        text.replace("\\n", " ")
            .replace("\\t", " ")
            .replace("\\r", " ")
            .replace("\\\\n", " ")
            .replace("\\\\t", " ")
            .replace("\\\\r", " ")
            .replace("\n", " ")
            .replace("\t", " ")
            .replace("\r", " ")
    )

    # Remove URLs and emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Expand contractions before stripping punctuation
    text = contractions.fix(text)

    # Keep letters, digits, apostrophes, and spaces
    text = re.sub(r"[^a-z0-9' ]", " ", text)

    # Remove stray apostrophes not attached to words
    text = re.sub(r"\s+'\s+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_review(text: str, stop_words: set) -> str:
    text = normalize_review_text(text)
    if not text:
        return ""

    tokens = word_tokenize(text)

    cleaned_tokens = []
    for tok in tokens:
        tok = tok.strip().lower()
        if not tok:
            continue

        # Keep negation words
        if tok in NEGATION_WORDS:
            cleaned_tokens.append(tok)
            continue

        # Keep digits if they may be meaningful (e.g., 24, 10, 2am)
        if tok.isdigit():
            cleaned_tokens.append(tok)
            continue

        # Remove stop words
        if tok in stop_words:
            continue

        # Remove very short junk, but keep 2+ char tokens
        if len(tok) < 2:
            continue

        # Remove tokens that are mostly apostrophes or non-informative
        if re.fullmatch(r"'+", tok):
            continue

        cleaned_tokens.append(tok)

    return " ".join(cleaned_tokens)


def choose_review_column(df):
    candidates = ["review", "text", "review_text", "content"]
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if "review" in col.lower() or "text" in col.lower():
            return col

    raise ValueError("Could not find a review/text column.")


def main():
    print("Building stop word set ...")
    STOP_WORDS = build_stop_words()
    print(f"  Stop word set size: {len(STOP_WORDS)}")

    print(f"Reading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, dtype=str, low_memory=False)
    print(f"  Shape: {df.shape}")

    review_col = choose_review_column(df)
    print(f"  Using review column: {review_col}")

    # Step 1: Strip b'...' wrappers
    print("Cleaning b'...' byte-string wrappers ...")
    df = df.apply(lambda col: col.map(clean_value) if col.dtype == object else col)

    # Step 2: Drop redundant rolled-up columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        print(f"Dropping redundant columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # Step 3: Expand dict columns
    for col, prefix in DICT_COLUMNS.items():
        if col in df.columns:
            print(f"Expanding '{col}' -> '{prefix}.*' ...")
            df = expand_dict_column(df, col, prefix)

    # Step 4: Deduplicate columns
    print("Deduplicating columns ...")
    df = dedup_columns(df)

    # Step 5: Normalize u'...' unicode string literals
    print("Normalizing u'...' unicode string literals ...")
    df = df.apply(lambda col: col.map(normalize_ustr) if col.dtype == object else col)

    # Step 6: Strip residual bare single-quote wrappers
    print("Stripping residual bare quote wrappers ...")
    df = df.apply(lambda col: col.map(strip_bare_quotes) if col.dtype == object else col)

    # Step 7: Normalize empty strings -> NaN
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    # Step 8: Auto-detect and convert boolean columns -> 0/1
    bool_cols = []
    for col in df.columns:
        series = df[col]
        if isinstance(series, pd.Series) and is_bool_col(series):
            bool_cols.append(col)
            df[col] = to_bool_int(series)
    print(f"Boolean columns converted ({len(bool_cols)}): {bool_cols}")

    # Step 9: Convert hours columns -> float (hours open)
    for col in HOURS_COLS:
        if col in df.columns:
            df[col] = df[col].apply(hours_open).fillna(0.0)
    print("Hours columns converted to float.")

    # Step 10: Fill remaining NaNs
    for col in df.columns:
        if col in bool_cols or col in HOURS_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("unknown")

    # Step 11: Preprocess review text
    print("Preprocessing review text ...")
    df["review_clean"] = df[review_col].apply(lambda t: preprocess_review(t, STOP_WORDS))

    # Optional diagnostic columns to help you inspect cleaning quality
    df["review_char_len"] = df[review_col].astype(str).str.len()
    df["review_clean_char_len"] = df["review_clean"].astype(str).str.len()
    df["review_clean_word_count"] = df["review_clean"].astype(str).str.split().apply(len)

    # Step 12: Write output
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone! Saved to: {OUTPUT_FILE}")
    print(f"  Final shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")

    print("\nSample cleaned reviews:")
    sample_cols = [review_col, "review_clean"]
    print(df[sample_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
