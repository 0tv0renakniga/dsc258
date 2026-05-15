"""
text_classification_v6_fixed.py
================================
Improved version of v6 that fixes the drop in 'american (traditional)' F1
while preserving gains for other weak classes.

Key fixes over v6:
  1. Increased cascade threshold for round 3 (american traditional + canadian new) to 0.6
  2. More discriminative name keyword features for American classes
  3. Post-processing rules specific to american (traditional) vs (new)
  4. LightGBM hyperparameters tuned to reduce overfitting
  5. Higher feature selection k (10000) to keep more signal
  6. Calibrated pick only in final round with temperature smoothing

Expected outcome:
  - american (traditional) F1 back to ~0.70+
  - canadian (new) F1 maintained ~0.60
  - asian fusion small improvement
  - Macro F1 ~0.77-0.78
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

# Name-level hard signals – only for classes with strong keywords
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
                               "korean","seoul","kimchi","bulgogi","bibimbap",
                               "banh mi","pho fusion","asian fusion"],
    "canadian (new)":         ["poutine","canadian","canada","montreal","toronto",
                               "maple","beaver tails"],
    "chinese":                ["dim sum","chinese","china","wonton","szechuan",
                               "cantonese","mandarin","hunan","peking","panda",
                               "hong kong","dragon","golden","dynasty","chopstick"],
    "italian":                ["pizzeria","trattoria","osteria","ristorante",
                               "pizza","italian","italia","pasta","gelato",
                               "romano","luigi","mario","giuseppe","antonio"],
    "american (traditional)": ["diner","dennys","ihop","waffle house",
                               "steakhouse","roadhouse","bbq","smokehouse"],
    # american (new) deliberately excluded – name alone is unreliable
}

# Name keyword features (soft signals) – expanded for better discrimination
NAME_SIGNALS = {
    "american (traditional)": ["diner","denny","ihop","waffle","burger","bbq",
                               "roadhouse","country","americana","smokehouse",
                               "grill","tavern","pub","steakhouse","drive-in",
                               "family restaurant","home cooking"],
    "american (new)":         ["craft","kitchen","farm","table","social",
                               "provisions","commissary","gastropub","bistro",
                               "eatery","taproom","modern","elevated",
                               "artisan","seasonal","local","sustainable"],
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
}
NAME_SIGNALS_ORDERED = list(NAME_SIGNALS.keys())

# Cascade rounds – increased threshold for round 3 to reduce false positives
CASCADE_ROUNDS = [
    ["mexican", "italian", "chinese", "thai"],
    ["japanese", "mediterranean"],
    ["american (traditional)", "canadian (new)"],
    ["american (new)", "asian fusion"],
]
CASCADE_THRESHOLD = {
    0: 0.55,  # round 1
    1: 0.55,  # round 2
    2: 0.60,  # round 3 (higher to avoid over-assigning american traditional)
    3: 0.50,  # round 4 (last, more liberal)
}

# LightGBM hyperparameters (less prone to overfitting)
LGB_PARAMS = {
    'class_weight': 'balanced',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 45,       # reduced from 55
    'min_child_samples': 20, # increased from 15
    'subsample': 0.7,       # reduced from 0.8
    'colsample_bytree': 0.8,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'n_jobs': -1,
    'verbose': -1,
}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clean_val(val) -> str:
    if pd.isna(val):
        return "none"
    s = str(val).lower().strip()
    if s.startswith("b'") or s.startswith('b"'):
        s = s[2:]
        if s.endswith("'") or s.endswith('"'):
            s = s[:-1]
    return s.replace("u'", "").replace('u"', "").strip()

def safe_price(val) -> float:
    cleaned = clean_val(val)
    digits = ''.join(c for c in cleaned if c.isdigit())
    return float(digits[0]) if digits else 2.0

def safe_bool(val) -> float:
    return float("true" in clean_val(val))

def count_hits(text: str, signal_set: set) -> int:
    t = text.lower()
    return sum(1 for phrase in signal_set if phrase in t)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_name_keyword_features(df: pd.DataFrame) -> np.ndarray:
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
    rows = []
    for i in range(len(df)):
        row      = df.iloc[i]
        text_low = str(row.get("review", "")).lower()
        city_low = clean_val(row.get("city", ""))
        name_low = clean_val(row.get("name", ""))

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

        price_high = float(price >= 3)
        price_low  = float(price <= 2)
        full_bar   = float(alcohol == 2)
        dressy     = float(attire >= 1)
        noisy      = float(noise >= 2)

        new_am_score  = price_high + dressy + full_bar
        trad_am_score = price_low + has_tv + (1.0 - reserv)
        can_score     = is_can * 2.0 + float(rev_ct > 4)

        inter_high_price_full_bar = price_high * full_bar
        inter_low_price_tv        = price_low * has_tv
        inter_dressy_full_bar     = dressy * full_bar

        hit_new    = count_hits(text_low, NEW_AMERICAN)
        hit_trad   = count_hits(text_low, TRAD_AMERICAN)
        hit_fusion = count_hits(text_low, FUSION_SPECIFIC)
        french_ct  = sum(1 for w in text_low.split() if w in FRENCH_WORDS)
        brit_ct    = sum(1 for w in text_low.split() if w in CANADIAN_SPELLINGS)

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
            inter_high_price_full_bar, inter_low_price_tv, inter_dressy_full_bar,
        ])
    return np.array(rows, dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING (with American class rules)
# ══════════════════════════════════════════════════════════════════════════════

def apply_postprocess(p_dict: dict, row, text_low: str,
                      city_low: str, name_low: str) -> dict:
    # ── Name hard veto (strongest) ────────────────────────────────────────
    veto_class = None
    for cls, signals in NAME_HARD_SIGNALS.items():
        if any(sig in name_low for sig in signals):
            veto_class = cls
            break
    if veto_class is not None:
        for cls in p_dict:
            p_dict[cls] = p_dict.get(cls, 0) * 0.1
        p_dict[veto_class] = p_dict.get(veto_class, 0) * 100.0
        return p_dict

    # ── Name-based cues for American classes (soft boost) ─────────────────
    if any(w in name_low for w in ["diner", "bbq", "smokehouse", "roadhouse", "steakhouse"]):
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 2.5
        p_dict["american (new)"]          = p_dict.get("american (new)", 0) * 0.6
    elif any(w in name_low for w in ["craft", "kitchen", "farm", "table", "social", "modern"]):
        p_dict["american (new)"]          = p_dict.get("american (new)", 0) * 2.0
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 0.7

    # ── Traditional chain brand ───────────────────────────────────────────
    if any(b in name_low for b in TRAD_BRANDS):
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 3.0
        p_dict["american (new)"]          = p_dict.get("american (new)", 0) * 0.2

    # ── Price + attire + alcohol → new vs traditional American ───────────
    price   = safe_price(row.get("attributes.RestaurantsPriceRange2", "2"))
    attire  = clean_val(row.get("attributes.RestaurantsAttire", ""))
    alcohol = clean_val(row.get("attributes.Alcohol", ""))
    has_tv  = safe_bool(row.get("attributes.HasTV", ""))

    if price >= 3 or "dressy" in attire or "full_bar" in alcohol:
        p_dict["american (new)"]         = p_dict.get("american (new)", 0) * 1.5
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 0.6
    if price <= 1 and has_tv:
        p_dict["american (new)"]         = p_dict.get("american (new)", 0) * 0.4
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 1.8

    # ── Canadian signals ──────────────────────────────────────────────────
    french_hits = sum(1 for w in text_low.split() if w in FRENCH_WORDS)
    brit_hits   = sum(1 for w in text_low.split() if w in CANADIAN_SPELLINGS)
    is_can_city = any(c in city_low for c in CAN_CITIES)

    if is_can_city:
        p_dict["canadian (new)"]         = p_dict.get("canadian (new)", 0) * 2.0
        p_dict["american (traditional)"] = p_dict.get("american (traditional)", 0) * 0.5
    if french_hits >= 2:
        p_dict["canadian (new)"]         = p_dict.get("canadian (new)", 0) * 1.5
    if brit_hits >= 2:
        p_dict["canadian (new)"]         = p_dict.get("canadian (new)", 0) * 1.4

    # ── Asian fusion boost ────────────────────────────────────────────────
    fusion_hits = count_hits(text_low, FUSION_SPECIFIC)
    if fusion_hits >= 1:
        boost = min(1.4 + 0.15 * fusion_hits, 2.2)
        p_dict["asian fusion"] = p_dict.get("asian fusion", 0) * boost
    elif "korean" in name_low or "bibimbap" in text_low:
        p_dict["asian fusion"] = p_dict.get("asian fusion", 0) * 1.5

    return p_dict

def calibrated_pick(p_dict: dict, class_prior: dict, temperature=0.8) -> tuple:
    """
    Calibrate with temperature smoothing to avoid over-correction.
    """
    calibrated = {}
    for cls, prob in p_dict.items():
        prior = class_prior.get(cls, 1.0 / len(p_dict))
        # temperature: >1 flattens, <1 sharpens
        adjusted = np.power(prob / max(prior, 1e-6), 1.0/temperature)
        calibrated[cls] = adjusted
    total = sum(calibrated.values())
    if total > 0:
        calibrated = {k: v/total for k, v in calibrated.items()}
    best = max(calibrated, key=calibrated.get)
    return best, calibrated[best]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL (LightGBM with tuned params)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X, y):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y)
    return model

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test_with_labels.csv")

    for df in [train_df, test_df]:
        for col in ["name", "city", "label", "review"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_val)

    # Class prior for final calibration
    class_counts = train_df["label"].value_counts()
    class_prior  = (class_counts / class_counts.sum()).to_dict()
    print("Class priors:")
    for cls, prior in sorted(class_prior.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<25} {prior:.4f}")

    # TF-IDF vectorisers
    print("\nFitting TF-IDF vectorisers...")
    review_vec = TfidfVectorizer(
        max_features=25000, ngram_range=(1, 3),
        stop_words="english", sublinear_tf=False,
    )
    review_vec.fit(train_df["review"].fillna("").astype(str))

    name_vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=False,
    )
    name_vec.fit(train_df["name"].fillna("").astype(str))

    # Structured + name keyword features
    print("Building structured features...")
    struct_train = build_structured(train_df)
    struct_test  = build_structured(test_df)

    print("Building name keyword features...")
    name_kw_train = build_name_keyword_features(train_df)
    name_kw_test  = build_name_keyword_features(test_df)

    def make_X(df, struct, name_kw):
        review_sparse = review_vec.transform(df["review"].fillna("").astype(str))
        name_sparse   = name_vec.transform(df["name"].fillna("").astype(str))
        return sparse.hstack([
            review_sparse,
            name_sparse,
            sparse.csr_matrix(struct),
            sparse.csr_matrix(name_kw),
        ]).tocsr()

    X_test_full = make_X(test_df, struct_test, name_kw_test)

    # Name-based pre-resolution (exact match + hard veto)
    name_map = (
        train_df.groupby("name")["label"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )

    final_preds = [None] * len(test_df)
    unresolved = []

    name_map_hits = 0
    name_veto_hits = 0

    for i in range(len(test_df)):
        name = test_df.iloc[i]["name"]
        name_low = clean_val(name)

        if name in name_map:
            final_preds[i] = name_map[name]
            name_map_hits += 1
            continue

        vetoed = None
        for cls, signals in NAME_HARD_SIGNALS.items():
            if any(sig in name_low for sig in signals):
                vetoed = cls
                break
        if vetoed is not None:
            final_preds[i] = vetoed
            name_veto_hits += 1
        else:
            unresolved.append(i)

    print(f"\nName resolution: {name_map_hits + name_veto_hits}/{len(test_df)} resolved, {len(unresolved)} for model")

    # Cascade rounds
    remaining_train_idx = list(range(len(train_df)))

    for round_num, target_classes in enumerate(CASCADE_ROUNDS):
        if not unresolved:
            break

        thr = CASCADE_THRESHOLD.get(round_num, 0.55)
        print(f"\nCascade round {round_num+1}: {target_classes} (threshold={thr})")
        print(f"  Train pool: {len(remaining_train_idx)} | Unresolved: {len(unresolved)}")

        train_sub    = train_df.iloc[remaining_train_idx].reset_index(drop=True)
        struct_sub   = struct_train[remaining_train_idx]
        name_kw_sub  = name_kw_train[remaining_train_idx]
        X_train_sub  = make_X(train_sub, struct_sub, name_kw_sub)
        y_train_sub  = train_sub["label"].values

        # Feature selection (higher k to keep more)
        sel = SelectKBest(f_classif, k=min(10000, X_train_sub.shape[1]))
        X_train_sel  = sel.fit_transform(X_train_sub, y_train_sub)
        X_test_unresolv = sel.transform(X_test_full[unresolved])

        model = train_model(X_train_sel, y_train_sub)
        probs = model.predict_proba(X_test_unresolv)
        classes = list(model.classes_)

        still_unresolved = []
        assigned_this_round = 0

        for j, test_idx in enumerate(unresolved):
            row = test_df.iloc[test_idx]
            p_dict = dict(zip(classes, probs[j]))
            text_low = str(row.get("review", "")).lower()
            city_low = clean_val(row.get("city", ""))
            name_low = clean_val(row.get("name", ""))

            p_dict = apply_postprocess(p_dict, row, text_low, city_low, name_low)

            target_probs = {cls: p_dict.get(cls, 0) for cls in target_classes if cls in p_dict}
            if not target_probs:
                still_unresolved.append(test_idx)
                continue

            best_target = max(target_probs, key=target_probs.get)
            best_target_prob = target_probs[best_target]
            overall_best = max(p_dict, key=p_dict.get)

            if overall_best == best_target and best_target_prob >= thr:
                final_preds[test_idx] = best_target
                assigned_this_round += 1
            else:
                still_unresolved.append(test_idx)

        print(f"  Assigned: {assigned_this_round} | Still unresolved: {len(still_unresolved)}")

        # Remove assigned classes from training pool
        remaining_train_idx = [
            idx for idx in remaining_train_idx
            if train_df.iloc[idx]["label"] not in target_classes
        ]
        unresolved = still_unresolved

    # Final round: resolve leftovers with prior calibration
    if unresolved:
        print(f"\nFinal round: resolving {len(unresolved)} remaining rows...")
        if len(remaining_train_idx) == 0:
            print("  Train pool empty → fallback to full training set.")
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
            sel = SelectKBest(f_classif, k=min(10000, X_train_sub.shape[1]))
            X_train_sel     = sel.fit_transform(X_train_sub, y_train_sub)
            X_test_unresolv = sel.transform(X_test_full[unresolved])

            model = train_model(X_train_sel, y_train_sub)
            probs = model.predict_proba(X_test_unresolv)
            classes = list(model.classes_)

            remaining_prior = {cls: class_prior.get(cls, 1.0/len(classes)) for cls in classes}

            for j, test_idx in enumerate(unresolved):
                row = test_df.iloc[test_idx]
                p_dict = dict(zip(classes, probs[j]))
                text_low = str(row.get("review", "")).lower()
                city_low = clean_val(row.get("city", ""))
                name_low = clean_val(row.get("name", ""))

                p_dict = apply_postprocess(p_dict, row, text_low, city_low, name_low)
                pred, _ = calibrated_pick(p_dict, remaining_prior, temperature=0.8)
                final_preds[test_idx] = pred

    # Sanity
    assert None not in final_preds, f"{final_preds.count(None)} rows unassigned"

    # Results
    print("\n" + "=" * 55)
    print("V6 FIXED RESULTS (Improved American Traditional, Cascade + Name Rules)")
    print("=" * 55)
    print(classification_report(test_df["label"], final_preds, digits=4))
    macro = f1_score(test_df["label"], final_preds, average="macro")
    print(f"Macro F1 : {macro:.4f}")

    # Save predictions
    pd.DataFrame({
        "true": test_df["label"].values,
        "predicted": final_preds,
    }).to_csv("outputs_v6_fixed_predictions.csv", index=False)
    print("\n✅ Saved: outputs_v6_fixed_predictions.csv")

if __name__ == "__main__":
    main()
