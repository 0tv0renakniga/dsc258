import pandas as pd

# 1. Load the two datasets
# csv1 = "test.csv"
# csv2 = "../kaggle/challenge/train.csv"
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('../kaggle/challenge/train.csv')

# 2. Create a subset of the training data with only the columns needed for the merge
# We use 'address' as the key and 'label' as the value to add.
# We also drop duplicates in case the same address appears multiple times in train.csv.
df_labels = df_train[['address', 'label']].drop_duplicates(subset=['address'])

# 3. Perform a left merge
# This keeps all rows in the test set and adds the label where addresses match.
result_df = pd.merge(df_test, df_labels, on='address', how='left')

# 4. Save the updated dataframe to a new CSV
result_df.to_csv('test_with_labels.csv', index=False)

print("Merge complete. The 'label' column has been added to the test data.")
