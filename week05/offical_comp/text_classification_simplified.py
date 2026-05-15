"""
cuisine_classifier_optimized.py
================================
Final High-Performance Pipeline
- Single-pass LightGBM (No complex cascades)
- TF-IDF with Unigrams, Bigrams, and Trigrams
- Structured Boolean Attributes (-1, 0, 1)
- Geographic (Lat/Long) Integration
- Class Balancing for Minority Cuisine Labels
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from scipy import sparse

warnings.filterwarnings("ignore")

def main():
    print("🚀 Loading cleaned data...")
    # Loading your clean datasets
    train_df = pd.read_csv("train_clean.csv")
    test_df  = pd.read_csv("test_clean.csv")

    # 1. TF-IDF TEXT FEATURES
    # Using 3000 features and strict min_df to filter noise like 'five'
    print("📝 Vectorizing Text (Names + Reviews)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
        min_df=5,
        max_df=0.5,
        stop_words='english'
    )
    
    # Anchor the review text with the name for maximum signal
    train_text = train_df["name"].astype(str) + " " + train_df["review_cleaned"].astype(str)
    test_text  = test_df["name"].astype(str) + " " + test_df["review_cleaned"].astype(str)
    
    X_train_tfidf = tfidf.fit_transform(train_text)
    X_test_tfidf  = tfidf.transform(test_text)
    master_attr_cols = [c for c in train_df.columns if c.startswith('attributes.')]
    print(f"📌 Using {len(master_attr_cols)} master attribute columns from training set.")

    # 2. UPDATED STRUCTURED MATRIX BUILDER
    def build_structured_matrix(df, master_cols):
        struct_data = {
            'stars': df['stars'].fillna(3.5),
            'rev_count': np.log1p(df['review_count'].fillna(0)),
            'lat': df['latitude'].fillna(0),
            'lon': df['longitude'].fillna(0)
        }
        
        # Force the dataframe to use the master column list
        for col in master_cols:
            if col in df.columns:
                struct_data[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
            else:
                # If a column exists in train but not test, fill with -1 (missing)
                struct_data[col] = np.full(len(df), -1)
            
        return pd.DataFrame(struct_data)
    
    X_train_struct = build_structured_matrix(train_df, master_attr_cols)
    X_test_struct  = build_structured_matrix(test_df, master_attr_cols)

    # 3. CONSOLIDATION
    # Stack TF-IDF (Sparse) with Structured (Converted to Sparse)
    X_train = sparse.hstack([X_train_tfidf, sparse.csr_matrix(X_train_struct.values)])
    X_test  = sparse.hstack([X_test_tfidf, sparse.csr_matrix(X_test_struct.values)])
    
    y_train = train_df["label"]
    y_test  = test_df["label"]

    # 4. LIGHTGBM MODEL
    # Reduced num_leaves and increased min_child_samples to handle gain warnings
    print("🌲 Training Balanced LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,          # Reduced from 70 to optimize gain and prevent overfitting
        min_child_samples=30,   # Ensures leaves are statistically significant
        class_weight="balanced", # Essential for Asian Fusion (2.7%) and Thai (3.6%)
        importance_type='gain',
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)

    # 5. FINAL EVALUATION
    preds = model.predict(X_test)
    
    print("\n" + "="*60)
    print("FINAL MODEL PERFORMANCE")
    print("="*60)
    print(classification_report(y_test, preds, digits=4))
    
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"OVERALL MACRO F1: {macro_f1:.4f}")

    # 6. SAVE PREDICTIONS
    output = pd.DataFrame({
        'name': test_df['name'],
        'actual': y_test,
        'predicted': preds
    })
    output.to_csv("final_cuisine_predictions.csv", index=False)
    print("\n✅ Results saved to: final_cuisine_predictions.csv")

if __name__ == "__main__":
    main()
