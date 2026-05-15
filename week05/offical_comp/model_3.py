import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train_df = pd.read_csv('train_cleaned.csv')
test_df = pd.read_csv('test_cleaned.csv')
truth_df = pd.read_csv('test_with_labels.csv')

# --- FORMATTING FIXES ---
truth_df.columns = truth_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()
if 'label' in test_df.columns:
    test_df = test_df.drop(columns=['label'])

# ==========================================
# 1. THE OMNI-TEXT GENERATOR
# ==========================================
def prepare_data(df):
    df = df.copy()
    for col in ['name', 'city', 'state', 'review']:
        df[col] = df[col].fillna('').astype(str)
    
    # We heavily weight the name and location by repeating them, ensuring CNB catches them
    df['omni_text'] = (df['name'] + " ") * 3 + (df['city'] + " ") * 2 + df['state'] + " " + df['review']
    return df

print("Preparing text features...")
train_df = prepare_data(train_df)
test_df = prepare_data(test_df)

X_train = train_df['omni_text']
y_train = train_df['label']
X_test = test_df['omni_text']

merged_test = pd.merge(test_df, truth_df[['id', 'label']], on='id')
y_test_true = merged_test['label']

# ==========================================
# 2. COMPLEMENT NAIVE BAYES PIPELINE
# ==========================================
print("Training Complement Naive Bayes (The Imbalanced Data Specialist)...")

model = Pipeline([
    # We use a massive vocabulary (75,000) to catch ultra-specific rare words 
    # ngram_range=(1,2) catches two-word phrases like "pad thai"
    ('tfidf', TfidfVectorizer(max_features=75000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)),
    
    # alpha=0.1 provides slight smoothing to handle words in the test set that weren't in the train set
    ('clf', ComplementNB(alpha=0.1))
])

# Train the model
model.fit(X_train, y_train)

# ==========================================
# 3. EXACT MATCH LOOKUP (The one hack we keep)
# ==========================================
# If we know the exact restaurant name from the training data, don't guess. Just use the answer.
name_to_label = train_df.groupby('name')['label'].agg(lambda x: x.mode()[0]).to_dict()

print("Generating predictions...")
test_predictions_ml = model.predict(X_test)

final_test_predictions = []
override_count = 0

for idx, row in test_df.iterrows():
    name = row['name']
    if name in name_to_label and name != '':
        final_test_predictions.append(name_to_label[name])
        override_count += 1
    else:
        final_test_predictions.append(test_predictions_ml[idx])

print(f"🔥 Hardcoded exact answers for {override_count} overlapping restaurants.")

# ==========================================
# 4. FINAL SCORE
# ==========================================
test_macro_f1 = f1_score(y_test_true, final_test_predictions, average='macro')
print(f"\n🚀 FINAL TEST MACRO F1 SCORE: {test_macro_f1:.4f}\n")

print("Full Test Classification Report:")
print("=" * 55)
print(classification_report(y_test_true, final_test_predictions, digits=4))

submission = pd.DataFrame({'id': test_df['id'], 'Predicted': final_test_predictions})
submission.to_csv('submission_final.csv', index=False)
print("✅ Successfully saved 'submission_final.csv'.")
