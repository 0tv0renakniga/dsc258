import pandas as pd

# Load
df_train = pd.read_csv("train_clean.csv")
df_test = pd.read_csv("test_cleaned.csv")

# Build dictionary: class -> unique training names
train_names = {
    label: set(df_train[df_train["label"] == label]["name"].dropna().astype(str).str.strip())
    for label in df_train["label"].unique()
}

# Reverse lookup: restaurant name -> class
name_to_label = {}
duplicate_names = set()

for label, names in train_names.items():
    for name in names:
        if name in name_to_label and name_to_label[name] != label:
            duplicate_names.add(name)
        else:
            name_to_label[name] = label

# Remove ambiguous names that appear in multiple classes
for name in duplicate_names:
    del name_to_label[name]

# Normalize test names
df_test["name_clean"] = df_test["name"].fillna("").astype(str).str.strip()

# Mark rows covered by dictionary
df_test["dict_label"] = df_test["name_clean"].map(name_to_label)
df_test["covered_by_dict"] = df_test["dict_label"].notna()

# Distribution of all test labels
print("\nFULL TEST DISTRIBUTION")
print(df_test["label"].value_counts().sort_index())

# Distribution of rows already covered by dictionary
print("\nCOVERED BY DICTIONARY")
print(df_test[df_test["covered_by_dict"]]["label"].value_counts().sort_index())

# Distribution of rows still needing prediction
remaining = df_test[~df_test["covered_by_dict"]].copy()

print("\nREMAINING TO PREDICT")
print(remaining["label"].value_counts().sort_index())

# Optional percentages
print("\nREMAINING TO PREDICT (%):")
print((remaining["label"].value_counts(normalize=True).sort_index() * 100).round(2))

print(f"\nTotal test rows: {len(df_test)}")
print(f"Covered by dictionary: {df_test['covered_by_dict'].sum()}")
print(f"Still need prediction: {len(remaining)}")
print(f"Coverage rate: {df_test['covered_by_dict'].mean():.4f}")
