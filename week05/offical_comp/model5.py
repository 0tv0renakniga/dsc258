"""
text_classification_v5_final.py
================================
Complete standalone pipeline. All fixes applied.

Key features:
  1. safe_price() / safe_bool() — handles b'2' byte-literal CSV artifacts
  2. Restaurant name TF-IDF + keyword features
  3. Name-based hard veto in post-processing
  4. Structured metadata (price, alcohol, attire, noise, stars)
  5. Cascade prediction: easy classes first, retrain on harder remainder
  6. Prior-calibrated argmax for final hard classes
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
from scipy import sparse

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

CAN_CITIES = [
    'ajax','aurora','brampton','calgary','mississauga','montreal','oakville',
    'pickering','richmond hill','scarborough','toronto','vaughan','whitby',
    'markham','etobicoke'
]
TRAD_BRANDS = [
    'burger king','five guys','wingstop','waffle bar','chick-fil-a',
    'mcdonald','white castle','dennys','denny','ihop','applebee',
    'cracker barrel','bob evans','perkins','waffle house',
]
NEW_BRANDS = [
    'farm','kitchen','craft','table','social','provisions',
    'commissary','collective','workshop','market',
]
FRENCH_WORDS = {
    'les','des','est','avec','pour','dans','qui','une','sur','par',
    'mais','pas','plus','nous','vous','ils','leur','tout','bien',
    'très','aussi','comme','quand','même','encore','toujours','jamais',
    'resto','était','bonne','bon','belle','beau','nourriture',
    'délicieux','excellent','parfait','vraiment','endroit',
}
CANADIAN_SPELLINGS = {
    'favourite','flavour','colour','behaviour','honour','neighbour',
    'labour','centre','theatre','licence','practise','programme',
}
"""
FUSION_SPECIFIC = {
    'korean bbq','bulgogi','bibimbap','kimchi','filipino','hawaiian',
    'poke bowl','poke','banh mi','bao bun','bao','loco moco',
    'spam musubi','kalbi','galbi','japchae','lumpia','adobo',
    'sisig','lechon','halo halo','korean fried chicken','kbbq',
}
NEW_AMERICAN = {
    'craft beer','craft cocktail','farm to table','farm-to-table',
    'seasonal menu','small plates','charcuterie','artisanal','artisan',
    'gastropub','gastrobar','speakeasy','rooftop','tasting menu',
    'house made','housemade','locally sourced','mixologist','sommelier',
    'truffle','foie gras','short rib','bone marrow','duck confit',
    'burrata','flatbread','bruschetta','elevated','upscale casual',
}
TRAD_AMERICAN = {
    'blue plate','comfort food','home cooking','homestyle','home-style',
    'country cooking','all day breakfast','all-day breakfast',
    'diner','greasy spoon','dive bar','sports bar','roadhouse',
    'country gravy','chicken fried steak','pot roast','meatloaf',
    'chicken pot pie','biscuits and gravy','early bird','senior discount',
}

# Name-level hard signals — if ANY of these appear in restaurant name,
# assign that class immediately without model (ordered by specificity)
NAME_HARD_SIGNALS = {
    "japanese":               ["sushi","ramen","izakaya","hibachi","yakitori",
                               "japanese","japan","tokyo","osaka","teriyaki",
                               "tempura","sake","udon","soba","katsu","miso"],
    "mexican":                ["taco","burrito","cantina","taqueria","tacos",
                               "mexican","mexico","enchilada","tamale","salsa",
                               "jalisco","guadalajara","tijuana","aztec","mariachi"],
    "thai":                   ["thai","thailand","bangkok","siam","pad thai"],
    "mediterranean":          ["falafel","shawarma","hummus","gyro","kebab",
                               "mediterranean","greek","lebanese","turkish",
                               "pita","souvlaki","tzatziki","baklava"],
    "asian fusion":           ["korean bbq","kbbq","poke","filipino","fusion",
                               "korean","seoul","kimchi","bulgogi","bibimbap"],
    "canadian (new)":         ["poutine"],
    "chinese":                ["dim sum","chinese","china","wonton","szechuan",
                               "cantonese","mandarin","hunan","peking","panda",
                               "hong kong","dragon","golden","dynasty","chopstick"],
    "italian":                ["pizzeria","trattoria","osteria","ristorante",
                               "pizza","italian","italia","pasta","gelato",
                               "romano","luigi","mario","giuseppe","antonio"],
    "american (traditional)": ["diner","dennys","ihop","waffle house",
                               "steakhouse","roadhouse","bbq","smokehouse"],
}
"""

FUSION_SPECIFIC = {
    'kimchi', 'poke bowl', 'poke', 'bulgogi', 'filipino', 'hawaiian',
    'bao', 'pork belly', 'bubble tea', 'vietnamese', 'fusion', 'korean',
    'seaweed', 'stir fry', 'cajun'  # Added from your importance list
}

NEW_AMERICAN = {
    'cocktail', 'kale', 'bisque', 'brunch', 'wine', 'seasonal',
    'craft beer', 'gastropub', 'small plates', 'farm to table',
}

TRAD_AMERICAN = {
    'burger', 'fry', 'wing', 'breakfast', 'diner', 'potato',
    'bacon', 'pancake', 'toast', 'steak', 'comfort food',
    'ribs', 'sports game', 'early bird' 
}

# Add these to NAME_HARD_SIGNALS specifically to help the "veto" logic
NAME_HARD_SIGNALS = {
    "japanese": ["sushi", "ramen", "miso", "tempura", "hibachi", "teriyaki", "sashimi", "ayce", "bento"],
    "mexican": ["taco", "burrito", "salsa", "tortilla", "carne asada", "guacamole", "chipotle"],
    "thai": ["pad thai", "curry", "green curry", "tom yum", "satay"],
    "mediterranean": ["shawarma", "falafel", "pita", "hummus", "gyro", "kabob", "baklava", "souvlaki"],
    "canadian (new)": ["poutine", "montreal", "maple"], 
    "chinese": ["fried rice", "dim sum", "wonton", "chow mein", "noodle", "panda express", "egg roll"],
    "asian fusion": ["poke","kimchi","hawaiian","bulgogi", "bubble tea"]
}
# Name keyword features (soft signals fed into model)
NAME_SIGNALS = {
    "japanese":               ["sushi","ramen","tokyo","japan","izakaya","sakura",
                               "kyoto","osaka","teriyaki","hibachi","miso","katsu"],
    "chinese":                ["china","chinese","panda","dragon","wok","dimsum",
                               "peking","szechuan","canton","hunan"],
    "italian":                ["pizza","italia","italian","trattoria","osteria",
                               "ristorante","romano","luigi","mario","tuscany"],
    "mexican":                ["mexico","mexican","taco","casa","cantina","burrito",
                               "hacienda","fiesta","jalisco"],
    "thai":                   ["thai","thailand","bangkok","siam","lotus","orchid"],
    "mediterranean":          ["greek","mediterranean","gyro","falafel","pita",
                               "hummus","kebab","istanbul","athens","beirut"],
    "asian fusion":           ["fusion","korean","seoul","kimchi","kbbq","filipino",
                               "hawaii","poke","banh","asian"],
    "canadian (new)":         ["poutine","canadian","canada","montreal","toronto",
                               "maple","beaver","alberta","ontario"],
    "american (new)":         ["craft","kitchen","farm","table","social",
                               "provisions","commissary","gastropub"],
    "american (traditional)": ["diner","denny","ihop","waffle","burger","bbq",
                               "roadhouse","country","americana"],
}
NAME_SIGNALS_ORDERED = list(NAME_SIGNALS.keys())

# Cascade rounds — predict best classes first, remove from pool, repeat
CASCADE_ROUNDS = [
    # ROUND 1: The "Identity" Classes (Massive unique signal > 0.07)
    ["mexican", "chinese", "thai", "japanese"], 
    
    # ROUND 2: High Certainty Specialty (Strong unique signal > 0.05)
    ["italian", "mediterranean"], 
    
    # ROUND 3: Regional & Solid Traditional (Lower score, but distinct)
    ["american (traditional)", "canadian (new)"], 
    
    # ROUND 4: The Residual Nuance (Hardest to distinguish, semantically broad)
    ["american (new)", "asian fusion"],
]
CASCADE_THRESHOLD = 0.55   # minimum confidence to accept a cascade assignment


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clean_val(val) -> str:
    if pd.isna(val): return "none"
    return str(val).lower().strip() # No more b' or u' stripping needed!

def safe_price(val) -> float:
    """Extract price 1-4 from any format including b'2'."""
    cleaned = clean_val(val)
    digits = ''.join(c for c in cleaned if c.isdigit())
    return float(digits[0]) if digits else 2.0

def safe_bool(val) -> float:
    # Your cleaned data likely uses 1/0 or True/False strings now
    s = str(val).lower()
    if s in ['1', '1.0', 'true']: return 1.0
    return 0.0

def count_hits(text: str, signal_set: set) -> int:
    """Count how many phrases from signal_set appear in text."""
    t = text.lower()
    return sum(1 for phrase in signal_set if phrase in t)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_name_keyword_features(df: pd.DataFrame) -> np.ndarray:
    """
    One column per cuisine class: count of name keyword hits.
    Restaurant names are short and highly predictive.
    e.g. 'Seoul Korean BBQ Fusion' → hits asian_fusion=2, korean=1
    """
    rows = []
    names = df.get("name", pd.Series("", index=df.index)).apply(clean_val)
    for name in names:
        row = [
            float(sum(1 for kw in NAME_SIGNALS[cls] if kw in name))
            for cls in NAME_SIGNALS_ORDERED
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def build_structured(df: pd.DataFrame) -> np.ndarray:
    """
    Structured metadata + explicit text signal counts.
    All attribute reads go through safe_price / safe_bool / clean_val
    so b'2' and similar artifacts never cause crashes.
    """
    rows = []
    for i in range(len(df)):
        row      = df.iloc[i]
        text_low = str(row.get("review", "")).lower()
        city_low = clean_val(row.get("city", ""))
        name_low = clean_val(row.get("name", ""))

        # ── Structured attributes ─────────────────────────────────────────
        price   = safe_price(row.get("attributes.RestaurantsPriceRange2", "2"))
        has_tv  = safe_bool(row.get("attributes.HasTV", ""))
        outdoor = safe_bool(row.get("attributes.OutdoorSeating", ""))
        reserv  = safe_bool(row.get("attributes.RestaurantsReservations", ""))
        groups  = safe_bool(row.get("attributes.RestaurantsGoodForGroups", ""))
        takeout = safe_bool(row.get("attributes.RestaurantsTakeOut", ""))
        kids    = safe_bool(row.get("attributes.GoodForKids", ""))
        delivery= safe_bool(row.get("attributes.RestaurantsDelivery", ""))
        happy   = safe_bool(row.get("attributes.HappyHour", ""))
        caters  = safe_bool(row.get("attributes.Caters", ""))
        wifi    = safe_bool(row.get("attributes.WiFi", ""))
        stars   = float(str(row.get("stars", 3.5)).strip() or 3.5)
        rev_ct  = np.log1p(float(str(row.get("review_count", 0)).strip() or 0))

        alcohol_raw = clean_val(row.get("attributes.Alcohol", ""))
        alcohol = (2.0 if "full_bar" in alcohol_raw or "full bar" in alcohol_raw
                   else 1.0 if "beer" in alcohol_raw or "wine" in alcohol_raw
                   else 0.0)

        attire_raw = clean_val(row.get("attributes.RestaurantsAttire", ""))
        attire = (2.0 if "formal" in attire_raw
                  else 1.0 if "dressy" in attire_raw
                  else 0.0)

        noise_raw = clean_val(row.get("attributes.NoiseLevel", ""))
        noise = (3.0 if "very_loud" in noise_raw
                 else 2.0 if "loud" in noise_raw
                 else 1.0 if "average" in noise_raw
                 else 0.0)

        is_can = float(any(c in city_low for c in CAN_CITIES))
        lat    = float(str(row.get("latitude",  0)).strip() or 0)
        lon    = float(str(row.get("longitude", 0)).strip() or 0)

        # ── Engineered interaction scores ─────────────────────────────────
        new_am_score  = float(price >= 3) + float(attire >= 1) + float(alcohol == 2)
        trad_am_score = float(price <= 2) + has_tv + (1.0 - reserv)
        can_score     = is_can * 2.0 + float(rev_ct > 4)

        # ── Explicit text signal counts ───────────────────────────────────
        hit_new    = count_hits(text_low, NEW_AMERICAN)
        hit_trad   = count_hits(text_low, TRAD_AMERICAN)
        hit_fusion = count_hits(text_low, FUSION_SPECIFIC)
        french_ct  = sum(1 for w in text_low.split() if w in FRENCH_WORDS)
        brit_ct    = sum(1 for w in text_low.split() if w in CANADIAN_SPELLINGS)

        # ── Name signals ──────────────────────────────────────────────────
        is_trad_brand = float(any(b in name_low for b in TRAD_BRANDS))
        is_new_brand  = float(any(w in name_low for w in NEW_BRANDS))
        is_asian_name = float(any(w in name_low for w in
                                  ["asian","korean","fusion","filipino","hawaiian"]))

        rows.append([
            price, has_tv, outdoor, reserv, groups, takeout, kids,
            delivery, happy, caters, wifi, stars, rev_ct,
            alcohol, attire, noise, is_can, lat, lon,
            new_am_score, trad_am_score, can_score,
            hit_new, hit_trad, hit_fusion, french_ct, brit_ct,
            is_trad_brand, is_new_brand, is_asian_name,
        ])
    return np.array(rows, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def apply_postprocess(p_dict: dict, row, text_low: str,
                      city_low: str, name_low: str) -> dict:
    """
    Domain rules applied on top of raw model probabilities.
    Returns modified p_dict — caller takes argmax.
    """
    # ── Name hard veto: strong cuisine keyword in restaurant name ─────────
    for cls, signals in NAME_HARD_SIGNALS.items():
        if any(sig in name_low for sig in signals):
            p_dict[cls] = p_dict.get(cls, 0) * 5.0
            break   # fire once only — most specific match wins

    # ── lat/lon signals ──────────────────────────────────────────────────
    lat = float(row.get("latitude", 0))
    lon = float(row.get("longitude", 0))
    
    # ── Known traditional chain brand ────────────────────────────────────
    if any(b in name_low for b in TRAD_BRANDS):
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 3.0
        p_dict["american (new)"]          = p_dict.get("american (new)", 0)         * 0.05

    # ── Price + attire + alcohol → new vs traditional American ───────────
    price   = safe_price(row.get("attributes.RestaurantsPriceRange2", "2"))
    attire  = clean_val(row.get("attributes.RestaurantsAttire", ""))
    alcohol = clean_val(row.get("attributes.Alcohol", ""))
    has_tv  = safe_bool(row.get("attributes.HasTV", ""))

    if price >= 3 or "dressy" in attire or "full_bar" in alcohol:
        p_dict["american (new)"]         = p_dict.get("american (new)", 0)         * 1.4
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 0.7
    if price <= 1 and has_tv:
        p_dict["american (new)"]         = p_dict.get("american (new)", 0)         * 0.4
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 1.5

    # ── Canadian signals ──────────────────────────────────────────────────
    french_hits = sum(1 for w in text_low.split() if w in FRENCH_WORDS)
    brit_hits   = sum(1 for w in text_low.split() if w in CANADIAN_SPELLINGS)
    is_can_city = any(c in city_low for c in CAN_CITIES)

    if is_can_city:
        p_dict["canadian (new)"]         = p_dict.get("canadian (new)", 0)         * 1.8
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 0.6
    if french_hits >= 3:
        p_dict["canadian (new)"]         = p_dict.get("canadian (new)", 0)         * 1.5
    if brit_hits >= 2:
        p_dict["canadian (new)"]         = p_dict.get("canadian (new)", 0)         * 1.3

    # ── Asian fusion boost ────────────────────────────────────────────────
    fusion_hits = count_hits(text_low, FUSION_SPECIFIC)
    if "kimchi" in text_low or "poke" in text_low:
        p_dict["asian fusion"] = p_dict.get("asian fusion", 0) * 2.0
    if fusion_hits >= 1:
        boost = min(1.3 + 0.2 * fusion_hits, 2.0)
        p_dict["asian fusion"] = p_dict.get("asian fusion", 0) * boost
    else:
        top_vals = sorted(p_dict.values())
        gap = (top_vals[-1] - top_vals[-2]) if len(top_vals) >= 2 else 1.0
        if gap < 0.15 and ("fusion" in text_low or "korean" in name_low):
            p_dict["asian fusion"] = p_dict.get("asian fusion", 0) * 1.4

    return p_dict


def calibrated_pick(p_dict: dict, class_prior: dict) -> tuple:
    """
    Divide each probability by its training prior then renormalise.
    Boosts minority classes (asian fusion, canadian) that argmax
    systematically under-predicts.
    Returns (best_class, calibrated_confidence).
    """
    calibrated = {
        cls: prob / class_prior.get(cls, 1.0)
        for cls, prob in p_dict.items()
    }
    total = sum(calibrated.values())
    calibrated = {k: v / total for k, v in calibrated.items()}
    best = max(calibrated, key=calibrated.get)
    return best, calibrated[best]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X, y):
    model = lgb.LGBMClassifier(
        class_weight="balanced",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=10,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    train_df = pd.read_csv("train_clean.csv")
    test_df  = pd.read_csv("test_clean.csv")

    for df in [train_df, test_df]:
        for col in ["name", "city", "label", "review"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_val)

    # ── Name history lookup ───────────────────────────────────────────────────
    name_map = (
        train_df.groupby("name")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )

    # ── Class prior for calibrated argmax ────────────────────────────────────
    class_counts = train_df["label"].value_counts()
    class_prior  = (class_counts / class_counts.sum()).to_dict()
    print("Class priors:")
    for cls, prior in sorted(class_prior.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<25} {prior:.4f}")

    # ── Fit TF-IDF vectorisers on full training set ───────────────────────────
    print("\nFitting TF-IDF vectorisers...")
    review_vec = TfidfVectorizer(
        max_features=3000, 
        ngram_range=(1, 3),
        stop_words="english",
        min_df=5,
        max_df=0.5,
        sublinear_tf=False,  # chi2 requires non-negative
    )
    review_vec.fit(train_df["review_cleaned"].fillna("").astype(str))

    name_vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=False,  # must stay False — chi2 needs non-negative
    )
    name_vec.fit(train_df["name"].fillna("").astype(str))

    # ── Build structured + name keyword features once ─────────────────────────
    print("Building structured features...")
    struct_train = build_structured(train_df)
    struct_test  = build_structured(test_df)

    print("Building name keyword features...")
    name_kw_train = build_name_keyword_features(train_df)
    name_kw_test  = build_name_keyword_features(test_df)

    def make_X(df, struct, name_kw):
        """Combine all feature sources into one sparse matrix."""
        review_sparse = review_vec.transform(df["review_cleaned"].fillna("").astype(str))
        name_sparse   = name_vec.transform(df["name"].fillna("").astype(str))
        return sparse.hstack([
            review_sparse,
            name_sparse,
            sparse.csr_matrix(struct),
            sparse.csr_matrix(name_kw),
        ])

    X_test_full = make_X(test_df, struct_test, name_kw_test)

    # ── Name-lookup pass ──────────────────────────────────────────────────────
    final_preds = [None] * len(test_df)
    unresolved  = []

    name_map_hits  = 0
    name_veto_hits = 0
    name_veto_log  = []   # (test_idx, name, vetoed_to_class)

    for i in range(len(test_df)):
        row      = test_df.iloc[i]
        name     = row["name"]
        name_low = clean_val(name)

        # Exact name-map match
        if name in name_map:
            final_preds[i] = name_map[name]
            name_map_hits += 1
            continue

        # Name hard-veto: strong cuisine keyword in restaurant name
        # If fired, we assign immediately without needing the model
        vetoed = None
        for cls, signals in NAME_HARD_SIGNALS.items():
            if any(sig in name_low for sig in signals):
                vetoed = cls
                break

        if vetoed is not None:
            final_preds[i] = vetoed
            name_veto_hits += 1
            name_veto_log.append((i, name_low, vetoed))
        else:
            unresolved.append(i)

    print(f"\nName resolution summary:")
    print(f"  Exact name-map match : {name_map_hits}")
    print(f"  Name hard-veto match : {name_veto_hits}")
    print(f"  Total resolved       : {name_map_hits + name_veto_hits} / {len(test_df)}")
    print(f"  Unresolved (→ model) : {len(unresolved)}")

    if name_veto_log:
        print(f"\n  Sample name-veto assignments (first 15):")
        print(f"  {'Name':<35} → {'Assigned class'}")
        print(f"  {'-'*60}")
        for _, nm, cls in name_veto_log[:15]:
            print(f"  {nm:<35} → {cls}")

    # ── Cascade rounds ────────────────────────────────────────────────────────
    remaining_train_idx = list(range(len(train_df)))

    for round_num, target_classes in enumerate(CASCADE_ROUNDS):
        if not unresolved:
            break

        print(f"\nCascade round {round_num+1}: {target_classes}")
        print(f"  Train pool: {len(remaining_train_idx)} | "
              f"Unresolved: {len(unresolved)}")

        train_sub    = train_df.iloc[remaining_train_idx].reset_index(drop=True)
        struct_sub   = struct_train[remaining_train_idx]
        name_kw_sub  = name_kw_train[remaining_train_idx]
        X_train_sub  = make_X(train_sub, struct_sub, name_kw_sub)
        y_train_sub  = train_sub["label"].values

        sel = SelectKBest(f_classif, k=min(8000, X_train_sub.shape[1]))
        X_train_sel  = sel.fit_transform(X_train_sub, y_train_sub)
        X_test_unresolv = sel.transform(X_test_full[unresolved])

        model   = train_model(X_train_sel, y_train_sub)
        probs   = model.predict_proba(X_test_unresolv)
        classes = list(model.classes_)

        still_unresolved   = []
        assigned_this_round = 0

        for j, test_idx in enumerate(unresolved):
            row      = test_df.iloc[test_idx]
            p_dict   = dict(zip(classes, probs[j]))
            text_low = str(row.get("review", "")).lower()
            city_low = clean_val(row.get("city", ""))
            name_low = clean_val(row.get("name", ""))

            p_dict = apply_postprocess(p_dict, row, text_low, city_low, name_low)

            # Only assign if top prediction is a target class above threshold
            target_probs = {cls: p_dict.get(cls, 0) for cls in target_classes
                            if cls in p_dict}
            if not target_probs:
                still_unresolved.append(test_idx)
                continue

            best_target      = max(target_probs, key=target_probs.get)
            best_target_prob = target_probs[best_target]
            overall_best     = max(p_dict, key=p_dict.get)

            # Dynamic threshold: lower it for Asian Fusion/American New
            current_thresh = 0.45 if ("asian fusion" in target_classes) else CASCADE_THRESHOLD

            if overall_best == best_target and best_target_prob >= CASCADE_THRESHOLD:
                final_preds[test_idx] = best_target
                assigned_this_round  += 1
            else:
                still_unresolved.append(test_idx)

        print(f"  Assigned: {assigned_this_round} | "
              f"Still unresolved: {len(still_unresolved)}")

        # Remove assigned classes from training pool for next round
        remaining_train_idx = [
            idx for idx in remaining_train_idx
            if train_df.iloc[idx]["label"] not in target_classes
        ]
        unresolved = still_unresolved

    # ── Final round: resolve anything remaining ───────────────────────────────
    if unresolved:
        print(f"\nFinal round: resolving {len(unresolved)} remaining rows...")
        # Pool is exhausted when all cascade classes consumed — use full training set
        if len(remaining_train_idx) == 0:
            print("  Train pool empty — falling back to full training set.")
            remaining_train_idx = list(range(len(train_df)))

        train_sub   = train_df.iloc[remaining_train_idx].reset_index(drop=True)
        struct_sub  = struct_train[remaining_train_idx]
        name_kw_sub = name_kw_train[remaining_train_idx]
        X_train_sub = make_X(train_sub, struct_sub, name_kw_sub)
        y_train_sub = train_sub["label"].values

        unique_classes = np.unique(y_train_sub)
        if len(unique_classes) == 1:
            for idx in unresolved:
                final_preds[idx] = unique_classes[0]
        else:
            sel = SelectKBest(f_classif, k=min(8000, X_train_sub.shape[1]))
            X_train_sel     = sel.fit_transform(X_train_sub, y_train_sub)
            X_test_unresolv = sel.transform(X_test_full[unresolved])

            model   = train_model(X_train_sel, y_train_sub)
            probs   = model.predict_proba(X_test_unresolv)
            classes = list(model.classes_)

            remaining_prior = {
                cls: class_prior.get(cls, 1.0 / len(classes))
                for cls in classes
            }

            for j, test_idx in enumerate(unresolved):
                row      = test_df.iloc[test_idx]
                p_dict   = dict(zip(classes, probs[j]))
                text_low = str(row.get("review", "")).lower()
                city_low = clean_val(row.get("city", ""))
                name_low = clean_val(row.get("name", ""))

                p_dict = apply_postprocess(p_dict, row, text_low, city_low, name_low)
                pred, _ = calibrated_pick(p_dict, remaining_prior)
                final_preds[test_idx] = pred

    # ── Sanity check ──────────────────────────────────────────────────────────
    assert None not in final_preds, \
        f"{final_preds.count(None)} test rows were never assigned a prediction!"

    # ── Results ───────────────────────────────────────────────────────────────
    baseline = {
        "american (new)":         0.5450,
        "american (traditional)": 0.7598,
        "asian fusion":           0.4020,
        "canadian (new)":         0.5565,
        "chinese":                0.9261,
        "italian":                0.9119,
        "japanese":               0.8827,
        "mediterranean":          0.8458,
        "mexican":                0.9604,
        "thai":                   0.8970,
    }

    print("\n" + "=" * 55)
    print("V5 FINAL RESULTS")
    print("=" * 55)
    print(classification_report(test_df["label"], final_preds, digits=4))

    macro = f1_score(test_df["label"], final_preds, average="macro")
    print(f"Macro F1 : {macro:.4f}  (previous best: 0.7687)")

    rep = classification_report(test_df["label"], final_preds, output_dict=True)
    print("\nPer-class delta vs previous best:")
    print(f"  {'Class':<25} {'Prev':>6}  {'V5':>6}  {'Δ':>7}")
    print(f"  {'-'*52}")
    for cls in sorted(baseline.keys()):
        if cls in rep:
            v5    = rep[cls]["f1-score"]
            base  = baseline[cls]
            delta = v5 - base
            arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
            print(f"  {cls:<25} {base:>6.3f}  {v5:>6.3f}  {arrow} {abs(delta):>5.3f}")

    pd.DataFrame({
        "true":      test_df["label"].values,
        "predicted": final_preds,
    }).to_csv("outputs_v5_predictions.csv", index=False)
    print("\n✅ Saved: outputs_v5_predictions.csv")


if __name__ == "__main__":
    main()
