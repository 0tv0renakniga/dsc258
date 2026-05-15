# Merging on 'id' guarantees your rows align perfectly, even if the files were shuffled.
# We use sklearn's classification_report because it instantly calculates the precision, 
# recall, and F1-score for every single category, plus your overall Macro F1.

import pandas as pd
from sklearn.metrics import classification_report

# 1. Load the ground truth and your predictions
truth_df = pd.read_csv('test_with_labels.csv')
preds_df = pd.read_csv('submission_with_lookup.csv')

# 2. Merge on 'id' to ensure perfect row alignment
# Assuming the ground truth column is 'label' and prediction column is 'Predicted'
merged_df = pd.merge(truth_df, preds_df, on='id')

# 3. Generate the report
print("Classification Report (F1 Score per Category):")
print("=" * 55)

report = classification_report(
    merged_df['label'], 
    merged_df['Predicted'], 
    digits=4 # Show 4 decimal places for precision
)

print(report)
