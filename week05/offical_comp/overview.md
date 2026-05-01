## Overview
### DSC 258R NLP Competition
* https://www.kaggle.com/competitions/ucsd-dsc-258-r-nlp-kaggle-competition-spring-2026/overview
* https://www.kaggle.com/t/beaa1f17a2694767a032eef6239fca3f
Predict the **type of restaurant** from a row of restaurant attributes and a free-text customer review. This is a 10-way classification task; the cuisine labels are:

* american (traditional)
* mexican
* italian
* chinese
* american (new)
* japanese
* mediterranean
* canadian (new)
* thai
* asian fusion

The class distribution is imbalanced — **american (traditional)** is the most frequent class, **asian fusion** the rarest.

---

## Data

| File | Rows | Notes |
| :--- | :--- | :--- |
| **train.csv** | 9,200 | 62 columns including `label`, free-text `review`, and 60 structured restaurant attributes (hours, ambience, alcohol, price range, location, etc.). |
| **test.csv** | 3,944 | Same schema as `train.csv` with the `label` column removed. |
| **sample_submission.csv** | 3,944 | Format example. |
| **baseline.ipynb** | — | Reference model: averaged Word2Vec embeddings + logistic regression on the `review` column only. |

The structured attribute columns contain many missing values and string-encoded JSON (e.g., `attributes.Ambience`). Some preprocessing is expected — exploratory data analysis is recommended before feature engineering.

---

## Baseline
The provided notebook uses **only** the `review` text column: it tokenizes, removes stopwords/punctuation, learns Word2Vec embeddings on the training reviews, averages those embeddings to a per-document vector, and trains a logistic regression classifier. You are free to use any subset (or all) of the available features and any modeling approach.

* baseline on public leaderboard: 0.596
* *note: This leaderboard is calculated with approximately 30% of the test data. The final results will be based on the other 70%, so the final standings may be different.*

---

## Submission Format
Submit a CSV with a header row and two columns:

```csv
id,Predicted
0,american (traditional)
1,mexican
...
```
* id matches the id column in test.csv. Predicted must be one of the 10 cuisine labels listed above (case-sensitive, matching the strings in train.csv exactly).

## Evaluation

Submissions are scored by **macro-averaged F1** across the 10 cuisine classes. Each class contributes equally to the final score regardless of how common it is in the data — so a model that does well on *american (traditional)* but poorly on *asian fusion* will be penalized. This is intentional: the data is imbalanced, and a model that just over-predicts the majority class will not score well.

* **Public Leaderboard:** Reflects macro-F1 on a **public subset** (~30% of test rows).
* **Private Leaderboard:** Final ranking is determined by macro-F1 on a **private subset** (the remaining ~70%) revealed when the competition closes.

If you compute F1 locally, use:
`sklearn.metrics.f1_score(y_true, y_pred, average='macro')`

> **Note:** sklearn's default `average='binary'` will error on this multi-class task.

## Dataset Description

* **train.csv**: The training set. The `id` column is a unique row key. The `label` column is the cuisine type to predict. The remaining columns are a mix of free text (a customer review), numerical features, and categorical features describing the restaurant. There are many missing values — start with some exploratory data analysis before feature engineering.
* **test.csv**: Same format as `train.csv` with the `label` column removed. Predict labels for these rows.
* **sample_submission.csv**: A correctly-formatted submission file. Replace the `Predicted` column with your model's predictions and submit.
* **Baseline notebook**: See the **Code** tab for a baseline notebook you can copy and edit: a logistic regression model trained on document vectors computed by averaging word embeddings of the words in each review. The baseline uses review text only — you are free to use any subset (or all) of the available features and any modeling approach, and you should aim for better performance.
