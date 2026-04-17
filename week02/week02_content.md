# Tokenization, Stemming, and Lemmatization

* **Summary:** In this week,will outline a general framework for text mining and natural language processing tasks, where representation learning is arguably the most important step. In this lecture, we will introduce the classical bag-of-words representation and show how to use it to conduct document classification, using sentiment analysis as an example.

* **Goals:** 
    * Describe the general framework for text mining and natural language processing tasks
    * Apply bag-of-words representations to conduct text classification
    * Describe/apply the every (implementation) detail of bag-of-words representation.

## Lecture Material
- lec01_bag_of_words_classifier
- lec02_bag_of_words_representations
- lec03_building_linear_classifiers

## Code Material
- week02_zipfs_law_tf_idf_logistic_regression.ipynb
- movie_data.csv

---

## Lecture Material: Bag of Words Classifier

* **Summary:** This lecture introduces a classical text classification workflow built around **bag-of-words features**, **linear classifiers**, and **standard evaluation procedures**. The main idea is that documents must be converted into fixed-length numeric vectors before a classifier can learn from them, and the representation choice strongly affects performance. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

* **Why this lecture matters:** In the NLP pipeline, **representation** is the bridge between raw text and machine learning. A bag-of-words classifier does not read language the way humans do; it learns from vocabulary-based numeric features such as counts, binary indicators, or tf-idf weights. [bbengfort.github](https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/)

* **Likely testing style:** Based on Week 1, the instructor may prefer **clean definitions, implementation distinctions, default behavior, and concrete consequences of design choices** rather than only broad summaries. That means this section should foreground terms like `CountVectorizer`, `TfidfVectorizer`, vocabulary, sparse matrix, train/dev/test split, leakage, and the meanings of accuracy, precision, recall, and F1. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

***

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [Bag of Words Classifier](#bag-of-words-classifier)
- [A Typical Text Mining Pipeline](#a-typical-text-mining-pipeline)
- [Sentiment Analysis as an Example](#sentiment-analysis-as-an-example)
- [Bag-of-Words Representation](#bag-of-words-representation)
- [Vectorization Rules](#vectorization-rules)
- [N-grams and TF-IDF](#n-grams-and-tf-idf)
- [Data Validation and Splitting](#data-validation-and-splitting)
- [Evaluation Metrics](#evaluation-metrics)
- [Likely Test Targets](#likely-test-targets)
- [Generalized Rules](#generalized-rules)
- [Glossary](#glossary)

***

## Course and Lecture Context

### Lecture Title
- **Bag of Words Classifier**

### Core Focus
- This lecture explains how to turn documents into numeric feature vectors and use those vectors for classification. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- The key theme is that **representation is not a side detail**; it determines what information the classifier can and cannot use. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### High-Level Goal
- Learn the standard pipeline for document classification:
  - Prepare text.
  - Convert text into features.
  - Train a classifier.
  - Evaluate predictions with appropriate metrics. [bbengfort.github](https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/)

### Week 2 Priority
- Compared with Week 1, this lecture shifts from **preprocessing mechanics** to **feature construction and supervised classification**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- The most testable ideas are likely to be the exact meaning of bag-of-words, vocabulary learning, tf-idf, data splitting, and evaluation metrics. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Bag of Words Classifier

### Definition
- A **bag-of-words classifier** represents each document as a vector indexed by vocabulary terms, then trains a model on those vectors to predict labels such as positive or negative sentiment. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### Core Mechanism
- Step 1: Build a vocabulary from training documents. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Step 2: Convert each document into a feature vector over that vocabulary. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Step 3: Train a classifier on the resulting document-term matrix. [bbengfort.github](https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/)

### Main Intuition
- The classifier does not directly understand meaning, syntax, or discourse. Instead, it learns statistical relationships between feature patterns and labels. [bbengfort.github](https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/)
- If certain words or phrases appear more often in positive documents than in negative ones, a linear model can use those patterns for prediction. [bbengfort.github](https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/)

### Important Limitation
- Bag-of-words usually ignores most word order, so it can miss deeper compositional meaning even when it works well in practice. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

***

## A Typical Text Mining Pipeline

### Pipeline Overview
\[
\text{Preprocessing} \rightarrow \text{Representation} \rightarrow \text{Model Training}
\]

- **Preprocessing:** Clean or normalize text so it is easier to handle computationally. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- **Representation:** Convert text into machine-usable numeric features such as counts or tf-idf weights. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- **Model Training:** Fit a predictive model for the target task. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Why representation matters
- The classifier only sees the feature matrix, not the original raw text. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Because of that, the representation determines what information is preserved, discarded, emphasized, or ignored. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### Main takeaway
- Better modeling does not always come from changing the classifier; it often comes from improving the representation. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Sentiment Analysis as an Example

### Task definition
- **Sentiment analysis** is a document classification task in which the model predicts whether text expresses positive or negative sentiment. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- A standard example is labeling movie reviews as positive (1) or negative (0). [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

### Why bag-of-words works here
- Sentiment is often correlated with repeated lexical patterns such as positive adjectives, negative adjectives, and short phrases like “not good.” [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Because those signals often appear at the word or short-phrase level, bag-of-words and n-gram features can be effective baselines. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### Important caution
- Bag-of-words may still miss nuanced context, sarcasm, or long-range dependencies. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

***

## Bag-of-Words Representation

### Definition
- **Bag-of-words** represents a document as a fixed-length vector whose dimensions correspond to vocabulary terms. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The feature value for each term may be a count, a binary indicator, or a weighted value such as tf-idf. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### What it keeps
- Whether a term appears. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- How often a term appears, if counts are used. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Short local order patterns, if n-grams are included. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### What it discards
- Most word order. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Most grammar and syntax. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Most long-distance contextual relationships. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Practical consequence
- Two documents with similar vocabulary distributions can look similar to the classifier even if their wording or deeper meaning differs. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- This simplicity makes bag-of-words efficient and useful, but it also limits semantic precision. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Vectorization Rules

### CountVectorizer
- `CountVectorizer` converts a collection of documents into a **document-term matrix** of token counts. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The result is typically a **sparse matrix** because each document uses only a small subset of the full vocabulary. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### Default behavior
- `CountVectorizer` lowercases text by default. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Its default tokenization pattern usually keeps tokens with two or more alphanumeric characters. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Under the default rules, punctuation is usually treated as a separator rather than a feature token. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Vocabulary learning
- The vocabulary is learned during `fit` on the training corpus. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- When new documents are transformed, unseen terms do not create new features; they are ignored. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- This means the feature space is fixed after fitting. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Frequency controls
- `min_df` removes terms that appear in too few documents. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- `max_df` removes terms that appear in too many documents and can behave like corpus-specific stopword filtering. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- `binary=True` converts counts into simple presence/absence features. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Testable distinction
- `fit` learns the vocabulary or weighting statistics. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- `transform` uses what was already learned and does **not** relearn from new data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- `fit_transform` does both on the same data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## N-grams and TF-IDF

### N-grams
- A **unigram** is one token. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- A **bigram** is a sequence of two adjacent tokens. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Higher-order n-grams preserve more local word order than plain unigram features. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### Why n-grams matter
- Unigrams usually lose order information almost completely. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Bigrams can preserve short patterns like `not good`, which may change meaning in classification tasks. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### TF-IDF
- **TF-IDF** stands for **term frequency-inverse document frequency**. [bogotobogo](https://www.bogotobogo.com/python/NLTK/tf_idf_with_scikit-learn_NLTK.php)
- It increases the influence of terms that are important in a document but not too common across the corpus. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- `TfidfVectorizer` is conceptually similar to applying count vectorization and then tf-idf reweighting. [topic-modeling.pythonhumanities](https://topic-modeling.pythonhumanities.com/02_03_setting_up_tf_idf.html)

### Why TF-IDF can help
- Raw counts can overvalue terms that appear frequently in many documents. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- TF-IDF often improves feature usefulness by downweighting overly common terms and emphasizing more discriminative ones. [bogotobogo](https://www.bogotobogo.com/python/NLTK/tf_idf_with_scikit-learn_NLTK.php)

### Common confusion
- TF-IDF is still a vocabulary-based representation. [bogotobogo](https://www.bogotobogo.com/python/NLTK/tf_idf_with_scikit-learn_NLTK.php)
- It is **not** the same thing as dense embeddings or deep contextual representations. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Data Validation and Splitting

### Train / Dev / Test
- **Train:** used to fit the model and learn representation parameters such as vocabulary and idf weights. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- **Validation / Dev:** used for model selection, tuning, or comparing alternatives. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- **Test:** used once at the end for final evaluation. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Leakage rule
- You must split the data **before** learning the vocabulary or tf-idf weights. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- If the vectorizer is fit on all data before splitting, information from validation or test documents leaks into the feature space. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- That leakage can make performance estimates look better than they really are. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### K-fold cross validation
- In **k-fold cross validation**, the training portion is divided into \(k\) folds, and the model is trained \(k\) times with a different held-out fold each time. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The scores across folds are aggregated to estimate performance more robustly. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Cross-validation is usually used for model selection on training data, not as a replacement for a final independent test set. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Important implementation idea
- Within each fold, the vectorizer should be fit only on that fold’s training partition, then applied to the held-out fold. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The same rule applies to tf-idf statistics and any learned preprocessing step. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Evaluation Metrics

### Confusion matrix terms
- **TP:** predicted positive, actually positive. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- **TN:** predicted negative, actually negative. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- **FP:** predicted positive, actually negative. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- **FN:** predicted negative, actually positive. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Accuracy
\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]
- Accuracy is the proportion of all predictions that are correct. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- It is easy to understand, but it can be misleading when classes are imbalanced. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Precision
\[
\text{Precision}=\frac{TP}{TP+FP}
\]
- Precision asks: among predicted positives, how many were truly positive? [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- It matters more when false positives are costly. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Recall
\[
\text{Recall}=\frac{TP}{TP+FN}
\]
- Recall asks: among actual positives, how many were found? [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- It matters more when false negatives are costly. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### F1 score
\[
F1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
\]
- F1 combines precision and recall into one score using the harmonic mean. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- It is useful when both types of error matter and one number is needed for comparison. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Metric selection rule
- There is no universally best metric. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The correct metric depends on class balance and the practical cost of FP versus FN errors. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Likely Test Targets

### High priority
- Define **bag-of-words** clearly and state what it preserves versus what it loses. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Know what `CountVectorizer` does and what its output represents. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Know the difference between counts, binary features, and tf-idf weights. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Know why bag-of-words matrices are usually sparse. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Know why the vectorizer must be fit on training data only. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Know the meanings and formulas of accuracy, precision, recall, and F1. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Medium priority
- Know what unigrams and bigrams are, and why bigrams can help with patterns like negation. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)
- Know what `min_df`, `max_df`, and `binary=True` do conceptually. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Know the role of cross-validation versus a final test set. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Lower priority unless emphasized in quiz or code
- Deep implementation details of sparse matrix internals. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Broader comparisons to neural embeddings unless the lecture explicitly tests them. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Generalized Rules

### Representation rules
- Bag-of-words is a family of vocabulary-based representations, not just one exact formula. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The same document can be represented with counts, binary indicators, or tf-idf weights depending on the modeling goal. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Vocabulary rules
- The vocabulary depends on tokenization, normalization, and frequency thresholds. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Once learned, the vocabulary fixes the feature dimensions for all future transformed documents. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Modeling rules
- A linear classifier on bag-of-words features learns statistical associations between feature patterns and labels. [bbengfort.github](https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/)
- It does not “understand” text in a human-like semantic sense. [bbengfort.github](https://bbengfort.github.io/2016/05/text-classification-nltk-sckit-learn/)

### Evaluation rules
- A good reported score depends on sound data splitting and leakage prevention, not just on the classifier choice. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Better metrics do not matter if the experimental setup is flawed. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Practical rules
- Split first, fit later. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Learn vocabulary only from training data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Use validation or cross-validation for tuning. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Reserve test data for final evaluation. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Glossary

### **Bag-of-words**
- A text representation that maps each document to a vocabulary-based feature vector and usually ignores most word order. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### **CountVectorizer**
- A scikit-learn tool that converts documents into a sparse matrix of token counts. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **TfidfVectorizer**
- A scikit-learn tool that converts documents into tf-idf-weighted feature vectors. [topic-modeling.pythonhumanities](https://topic-modeling.pythonhumanities.com/02_03_setting_up_tf_idf.html)

### **Vocabulary**
- The set of feature terms learned from training documents. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Document-Term Matrix**
- A matrix where rows are documents, columns are vocabulary terms, and entries are counts or weights. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Sparse Matrix**
- A matrix in which most values are zero, which is common for text features. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### **Unigram**
- A one-token feature. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Bigram**
- A two-token adjacent sequence used as a feature. [pages.github.rpi](https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html)

### **Binary Feature**
- A feature that records only whether a term appears, not how many times it appears. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **TF-IDF**
- A term-weighting scheme that emphasizes terms important to a document but less common across the corpus. [bogotobogo](https://www.bogotobogo.com/python/NLTK/tf_idf_with_scikit-learn_NLTK.php)

### **Leakage**
- Accidental use of non-training information during fitting, which can inflate evaluation results. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Cross-validation**
- Repeated train/validation splitting used to estimate performance more robustly on training data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Precision**
- The fraction of predicted positives that are truly positive. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Recall**
- The fraction of actual positives that are correctly identified. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **F1 Score**
- The harmonic mean of precision and recall. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Accuracy**
- The fraction of all predictions that are correct. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Best improvements from your original version

The main upgrades are:
- stronger emphasis on **fit vs transform**, **training-only vocabulary learning**, and **leakage**, because those are highly testable implementation ideas. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- sharper distinction between **counts**, **binary features**, **n-grams**, and **tf-idf**, which are exactly the kinds of “what does this do?” concepts that often show up in quizzes. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- a dedicated **Likely Test Targets** section, since Week 1 suggests your instructor likes direct definitional and operational questions. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

If you want, send the next Week 2 section — probably **Bag of Words Representations** — and I’ll keep rewriting it in the same calibrated style so the whole week stays consistent before Quiz 2. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

---

Here’s a tighter, more quiz-oriented rewrite of **Lecture Material: Bag of Words Representations**. The main improvements are: sharper separation of **assumption vs implementation**, clearer formulas and what they mean, stronger connection to **Week 1’s “term definition” theme**, and more emphasis on distinctions the instructor is likely to test, such as **binary vs TF vs TF-IDF**, **document frequency vs term frequency**, and **why sparsity is unavoidable in text features**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

## Lecture Material: Bag of Words Representations

* **Summary:** This lecture explains how documents are converted into feature vectors under the **bag-of-words assumption**. The two main design choices are **what counts as a term** and **how each term is weighted**, and those choices determine what information the model keeps, what it ignores, and how useful the resulting representation is for classification. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

* **Why this lecture matters:** Week 1 focused on preprocessing and term definition, while this lecture shows how those decisions become actual vector features. In practice, bag-of-words is not one single formula but a family of representations built from a vocabulary plus a weighting scheme such as **binary**, **TF**, or **TF-IDF**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

* **Likely testing style:** Based on Quiz 1, expect questions that ask for **clean definitions, formula meanings, what each weighting scheme keeps or loses, and the difference between document-level importance and corpus-level rarity**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

***

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [Bag of Words Assumption](#bag-of-words-assumption)
- [Terms and Vocabulary](#terms-and-vocabulary)
- [Weighting Schemes](#weighting-schemes)
- [Binary Bag-of-Words](#binary-bag-of-words)
- [Term Frequency (TF) Variants](#term-frequency-tf-variants)
- [Inverse Document Frequency (IDF)](#inverse-document-frequency-idf)
- [TF-IDF Weighting and Zipf's Law](#tf-idf-weighting-and-zipfs-law)
- [Sparse Representation](#sparse-representation)
- [Likely Test Targets](#likely-test-targets)
- [General Rules](#general-rules)
- [Summary of Bag of Words](#summary-of-bag-of-words)
- [Glossary](#glossary)

***

## Course and Lecture Context

### Lecture Title
- **Bag of Words Representations**

### Core focus
- This lecture explains how to represent each document as a fixed-length numeric vector over a vocabulary. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- The central idea is that representation depends on two modeling choices: **term definition** and **term weighting**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Main goal
- Understand how the bag-of-words assumption produces a feature space for machine learning. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Compare the main weighting schemes and understand when each one is useful. [geeksforgeeks](https://www.geeksforgeeks.org/nlp/bag-of-words-vs-tf-idf/)

### Bridge from Week 1
- Week 1 asked, “What is a term?” and introduced tokenization, stemming, lemmatization, phrase mining, and Zipf’s law. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- This lecture extends that idea by showing that once terms are chosen, every document can be mapped into a vector whose entries are weights for those terms. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

***

## Bag of Words Assumption

### Core assumption
- The **bag-of-words assumption** is that a document can be represented by its terms and their weights while mostly ignoring word order. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- In other words, the model treats text as a collection of terms rather than a sequence with syntax and long-range structure. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### What this preserves
- Which terms appear. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- How often they appear, if count-based weights are used. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Some short local order, if n-grams are included instead of only single words. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### What this loses
- Most sentence structure and grammar. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Most long-distance dependencies and compositional meaning. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Important contrasts that depend on order, unless the chosen features explicitly encode them. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Key consequence
- Two documents can get similar vectors if they contain similar terms, even if their exact phrasing differs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- This simplification often works well for classification, but it cannot fully model nuance, negation, or syntax-sensitive meaning. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

***

## Terms and Vocabulary

### What is a term?
- A **term** is the basic feature unit used in the vector representation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Depending on preprocessing and modeling choices, a term can be a word, stem, lemma, phrase, or n-gram. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### What is a vocabulary?
- The **vocabulary** is the set of all terms used as feature dimensions. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- If the vocabulary is \(V=\{v_1,v_2,\dots,v_n\}\), then each document is represented as a vector of length \(n\). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Why vocabulary matters
- The vocabulary is not neutral; it depends on tokenization, normalization, stop-word handling, phrase construction, and frequency filtering. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Changing the vocabulary changes the feature space, which can change model behavior and performance. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Important rule
- If a term is not in the vocabulary, it contributes nothing to the final vector. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- This means vocabulary design is part of modeling, not just preprocessing cleanup. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

***

## Weighting Schemes

### Core question
- Once terms are defined, the next question is: **how much weight should each term receive in each document?** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Main options
- **Binary:** records whether the term appears. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- **TF:** records how often the term appears in the document. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- **TF-IDF:** combines document-level frequency with corpus-level rarity. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Main principle
- A weighting scheme should give larger values to terms that are more informative for distinguishing documents. [geeksforgeeks](https://www.geeksforgeeks.org/nlp/bag-of-words-vs-tf-idf/)
- Different tasks may prefer different weighting rules, so weighting is a modeling choice rather than a fixed standard. [geeksforgeeks](https://www.geeksforgeeks.org/nlp/bag-of-words-vs-tf-idf/)

### Testable distinction
- **Binary** asks “Did it appear?” [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- **TF** asks “How often did it appear in this document?” [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- **IDF** asks “How rare is it across the corpus?” [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- **TF-IDF** combines the last two ideas. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

***

## Binary Bag-of-Words

### Definition
- In **binary bag-of-words**, each feature is 1 if the term appears at least once and 0 otherwise. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

\[
x_{d,i}=
\begin{cases}
1 & \text{if term } v_i \text{ appears in document } d \\
0 & \text{otherwise}
\end{cases}
\]

### Interpretation
- Binary weighting captures **presence/absence** only. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Repetition after the first occurrence does not increase the feature value. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### When it helps
- Binary features are useful when occurrence matters more than repetition. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- They can also be more robust than raw counts for short texts or repetitive texts where repeated words add little new information. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Limitation
- Binary weighting throws away intensity information, so it cannot distinguish “appears once” from “appears twenty times.” [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## Term Frequency (TF) Variants

### Basic count form
- A simple count-based bag-of-words feature records how many times a term occurs in a document. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

\[
x_{d,i}=|\{j \mid w_j=v_i\}|
\]

### Core idea
- If a term appears more often in a document, that may indicate stronger relevance to that document. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Let \(f(t,d)\) denote the raw count of term \(t\) in document \(d\). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Common variants
- **Raw TF:** \(TF(t,d)=f(t,d)\) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- **Log TF:** \(TF(t,d)=\log(f(t,d)+1)\) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- **Maximum-frequency normalization:** scales counts relative to the most frequent term in the document. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- **Length-normalized or BM25-style TF:** reduces the advantage long documents get from simply having more words. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Why variants exist
- Repeating a word many times does not usually make it proportionally more informative. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Raw counts can bias the model toward longer or more repetitive documents. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Normalized TF variants try to preserve useful repetition while reducing document-length bias. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### General rule
- If document lengths vary a lot, normalized TF is usually safer than raw counts. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- If repetition itself is meaningful, count-based TF may still help. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## Inverse Document Frequency (IDF)

### Core concept
- **IDF** measures how informative a term is across the full corpus by looking at how many documents contain it. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Let \(df_D(t)\) be the number of documents in corpus \(D\) that contain term \(t\). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Intuition
- A term that appears in almost every document is usually less useful for distinguishing documents. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- A term that appears only in some documents is often more discriminative. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Typical form
- A common form is logarithmic, such as \(IDF(t)=\log\left(\frac{N}{df(t)}\right)\) or a smoothed variant. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- In scikit-learn-style explanations, idf reduces the weight of corpus-common terms and boosts the weight of rarer ones. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Important distinction
- **Term frequency** is document-specific. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- **Document frequency** is corpus-wide. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Students often confuse these, so keep them separate. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Caution
- Very rare terms may get large IDF values, but rarity alone does not guarantee usefulness. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Rare misspellings, noise, or one-off tokens can be highly weighted but still unhelpful. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## TF-IDF Weighting and Zipf's Law

### TF-IDF definition
- **TF-IDF** combines term frequency and inverse document frequency. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

\[
Weight(t,d)=TF(t,d)\times IDF(t)
\]

### Interpretation
- TF captures how important a term is within one document. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- IDF captures how discriminative that term is across the corpus. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- TF-IDF assigns large weights to terms that are common in a document but not common everywhere. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Why TF-IDF is popular
- It often works better than raw counts when frequent generic words dominate the corpus. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- It is still simple, interpretable, and compatible with sparse linear models. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Zipf connection
- Zipf’s law says word frequency is inversely related to rank, so a few words are extremely common and many words are very rare. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- This helps explain why both extremes can be problematic: very common words often carry little discriminative value, while very rare words may be sparse or noisy. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### Practical takeaway
- Mid-frequency content words are often more useful than either stopwords or one-off rare tokens. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- TF-IDF helps rebalance the feature space so that common words do not dominate solely because of frequency. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## Sparse Representation

### Why sparsity happens
- The vocabulary in text mining is usually large, but each document uses only a small subset of those terms. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- That means most entries in a document vector are zero. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### What sparse means
- A **sparse vector** or **sparse matrix** stores mainly the non-zero values instead of every zero explicitly. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- This makes large bag-of-words feature spaces computationally feasible. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Why this matters
- Without sparse storage, bag-of-words matrices can become too large to handle efficiently. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Many classical linear text models are especially effective with high-dimensional sparse inputs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Key rule
- Sparsity is not an accident in bag-of-words; it is a normal structural property of text features. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## Likely Test Targets

### High priority
- Define the **bag-of-words assumption** clearly. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Distinguish **term**, **vocabulary**, **TF**, **DF**, **IDF**, and **TF-IDF**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Explain what binary weighting captures versus what TF captures. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Explain why very common words get lower value under IDF. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Explain why bag-of-words vectors are sparse. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Medium priority
- Know why TF variants exist and what problem normalization is trying to solve. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Know the conceptual link between Zipf’s law and term filtering or reweighting. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- Know that vocabulary design depends on preprocessing choices from Week 1. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Likely quiz traps
- Confusing **term frequency** with **document frequency**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Treating TF-IDF as if it preserves syntax or deep semantics. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Forgetting that binary bag-of-words ignores repeated occurrences after the first. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Assuming rare terms are always useful just because their IDF is high. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## General Rules

### Representation design
- A bag-of-words representation depends on both **term definition** and **weight definition**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Those two design choices strongly affect interpretability, sparsity, and classification performance. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Frequency effects
- Very common words are often weak discriminators. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Very rare words may be noisy or unreliable. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Filtering and weighting help manage both extremes. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Independence simplification
- Bag-of-words effectively treats terms as mostly independent features once vectorized. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- This is useful computationally, but it ignores many interactions among words and phrases. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Task dependence
- No single weighting scheme is always best. [geeksforgeeks](https://www.geeksforgeeks.org/nlp/bag-of-words-vs-tf-idf/)
- Binary, TF, and TF-IDF should be viewed as design options matched to the task and corpus. [geeksforgeeks](https://www.geeksforgeeks.org/nlp/bag-of-words-vs-tf-idf/)

### Practical rules
- Use **binary** when presence matters more than repetition. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Use **TF** when repetition carries signal. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Use **TF-IDF** when common words need to be downweighted. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Expect tradeoffs between simplicity, interpretability, and expressive power. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## Summary of Bag of Words

### Strengths
- Simple and intuitive. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Easy to implement at scale. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- Works well with sparse linear models and provides a strong baseline for text classification. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### Weaknesses
- Ignores most syntax and word order. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Can struggle with negation, compositional meaning, phrase-level semantics, and word sense ambiguity. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Requires careful choices about preprocessing, vocabulary design, and weighting. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Best mental model
- Bag-of-words is not “the model understands language.” [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- It is “the model sees a weighted vocabulary footprint of each document.” [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## Glossary

### **Bag-of-words Assumption**
- The assumption that a document can be represented by its terms and weights while mostly ignoring word order. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### **Term**
- A textual unit used as a feature, such as a word, stem, lemma, phrase, or n-gram. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### **Vocabulary**
- The set of feature terms used as dimensions in the vector representation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### **Binary Bag-of-Words**
- A representation in which each feature records only whether a term appears. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### **Term Frequency (TF)**
- A document-level weighting based on how often a term occurs in one document. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### **Document Frequency (DF)**
- The number of documents in the corpus that contain a term. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### **Inverse Document Frequency (IDF)**
- A corpus-level weighting factor that increases the importance of terms appearing in fewer documents. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### **TF-IDF**
- A weighting scheme that combines document-level frequency with corpus-level rarity. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### **Sparse Vector/Matrix**
- A representation that stores mainly non-zero values because most entries are zero. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

### **Zipf’s Law**
- The empirical pattern that word frequency decreases roughly as rank increases, explaining why text contains both extremely common and very rare words. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

***

## What I improved

The biggest upgrades from your draft are:
- making **TF vs DF vs IDF** much more explicit, because that is a classic quiz target. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- tying the section directly back to **Week 1 term-definition choices**, so the course narrative stays coherent. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- adding **Likely Test Targets** and **quiz traps**, which fits the instructor pattern suggested by Quiz 1. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

Send the next Week 2 section whenever you want, and I’ll keep this same calibrated style so the whole study guide stays consistent. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

---

Here’s an improved, more test-ready rewrite of **Lecture Material: Building Linear Classifiers**. The main upgrade is that it now emphasizes the distinctions most likely to be quizzed: **linear score vs transformed output**, **regression vs classification**, **square loss vs log loss**, **why logistic regression fits text classification well**, and **how thresholds affect evaluation without changing the model itself**. Those are the kinds of clean conceptual contrasts that fit the Week 1 quiz style and naturally follow Week 2’s bag-of-words material. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

## Lecture Material: Building Linear Classifiers

* **Summary:** This lecture explains how numeric text representations, such as bag-of-words or tf-idf vectors, are turned into predictions using **linear models**. The core progression is: compute a linear score from input features, choose an output interpretation appropriate to the task, and train the model with a loss function that matches that task. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

* **Why this lecture matters:** After building representations, the next question is how to map those vectors to labels. In Week 2, the key comparison is between **Ordinary Least Squares (OLS)** for continuous regression targets and **Logistic Regression** for binary classification, especially for sparse text data like sentiment analysis. [kb.osu](https://kb.osu.edu/bitstreams/5c7cd0a0-48b1-5fa3-9e62-b0c9b658105a/download)

* **Likely testing style:** Expect questions that ask you to distinguish **linear prediction from logistic prediction**, **square loss from log loss**, **real-valued outputs from probability outputs**, and **probability thresholds from the learned model itself**. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

***

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [Building Linear Classifiers](#building-linear-classifiers)
- [Linear Prediction](#linear-prediction)
- [Ordinary Least Square (OLS) Regression](#ordinary-least-square-ols-regression)
- [From OLS to Logistic Regression](#from-ols-to-logistic-regression)
- [Logistic Regression for Classification](#logistic-regression-for-classification)
- [Using Logistic Regression for Bounded Regression](#using-logistic-regression-for-bounded-regression)
- [Logistic Regression for Sentiment Analysis](#logistic-regression-for-sentiment-analysis)
- [Likely Test Targets](#likely-test-targets)
- [General Rules](#general-rules)
- [Glossary](#glossary)

***

## Course and Lecture Context

### Lecture Title
- **Building Linear Classifiers**

### Core focus
- This lecture moves from **representation** to **prediction**. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Once each document has been converted to a feature vector, a linear model can assign weights to features and use those weights to predict a target value or class. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Main goal
- Understand how a weighted sum of features becomes a prediction. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Distinguish when that score should be interpreted as an unbounded numeric output versus a probability-like output for classification. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Week 2 connection
- Bag-of-words and tf-idf define the feature space. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Linear models learn how each feature contributes to the final prediction in that space. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

***

## Building Linear Classifiers

### Main idea
- A linear model computes a weighted combination of input features and then uses that score for prediction. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- The same basic linear score can support different tasks depending on the output transformation and loss function. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Prediction intuition
- A large positive weight means that feature pushes the score upward. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- A large negative weight means that feature pushes the score downward. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- The bias or intercept shifts the prediction even when all feature values are zero. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Why this works for text
- Text feature spaces are often high-dimensional and sparse, which makes linear methods attractive computationally. [web.stanford](https://web.stanford.edu/~jurafsky/slp3/4.pdf)
- In many text classification problems, useful signal is distributed across many vocabulary features rather than a small number of dense variables. [web.stanford](https://web.stanford.edu/~jurafsky/slp3/4.pdf)

***

## Linear Prediction

### General form
\[
\hat{y}_{i}=\sum_{j=1}^{d} x_{i,j} w_j + b
\]

- \(x_i\) is the feature vector for instance \(i\). [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- \(w_j\) is the learned weight for feature \(j\). [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- \(b\) is the intercept or bias term. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- \(\hat{y}_i\) is the model’s score or prediction. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Interpretation
- Each feature contributes additively according to its value and weight. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- In sparse text data, most feature values are zero, so only a small subset of terms contributes for any one document. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

### Key distinction
- A linear score by itself is just a numeric quantity. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Whether it is used for regression or classification depends on how the score is interpreted and what loss function is optimized. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

***

## Ordinary Least Square (OLS) Regression

### Purpose
- **Ordinary Least Squares** is a linear regression method used when the target variable is continuous. [kb.osu](https://kb.osu.edu/bitstreams/5c7cd0a0-48b1-5fa3-9e62-b0c9b658105a/download)
- It predicts an unbounded real-valued output from the linear score. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Prediction rule
\[
\hat{y}_{i}=\sum_{j=1}^{d}x_{i,j}w_{j}+b
\]

### Square loss
- For one instance: [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
\[
l(y_i,\hat{y}_i)=(y_i-\hat{y}_i)^2
\]

- For all training instances: [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
\[
L=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

### Interpretation
- OLS chooses weights that minimize the total squared error. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Squaring means large errors are penalized more heavily than small errors. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Because of that, OLS can be sensitive to outliers or unusually large mistakes. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Important rule
- OLS is appropriate when the target is numeric and unbounded or naturally continuous. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- It is not a probability model, and its outputs are not constrained to lie between 0 and 1. [kb.osu](https://kb.osu.edu/bitstreams/5c7cd0a0-48b1-5fa3-9e62-b0c9b658105a/download)

***

## From OLS to Logistic Regression

### Why OLS is not ideal for binary classification
- In binary classification, the label is typically 0 or 1. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- A plain linear model can output any real number, including values below 0 or above 1. [kb.osu](https://kb.osu.edu/bitstreams/5c7cd0a0-48b1-5fa3-9e62-b0c9b658105a/download)
- That makes OLS awkward for predicting class probabilities. [kb.osu](https://kb.osu.edu/bitstreams/5c7cd0a0-48b1-5fa3-9e62-b0c9b658105a/download)

### Logistic idea
- Logistic Regression keeps the linear score but applies a **sigmoid** transformation to map that score into the interval \((0,1)\). [web.stanford](https://web.stanford.edu/~jurafsky/slp3/4.pdf)

\[
\hat{y}_{i}=\sigma\left(\sum_{j=1}^{d}x_{i,j}w_j+b\right)
\]

### Sigmoid function
\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

### Interpretation
- Large positive scores map to values near 1. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Large negative scores map to values near 0. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- A score near 0 maps to a probability near 0.5. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Key insight
- Logistic Regression is still linear in the feature score, but non-linear in the output mapping. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- That is why it remains a linear classifier even though the final output is passed through a sigmoid. [web.stanford](https://web.stanford.edu/~jurafsky/slp3/4.pdf)

***

## Logistic Regression for Classification

### Target and output
- In binary Logistic Regression, the observed target is \(y_i \in \{0,1\}\). [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- The model output is a probability-like value in \([0,1]\) for the positive class. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Log loss
\[
l(y_i,\hat{y}_i)=-(y_i\log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i))
\]

### Interpretation
- The loss is small when the model assigns high probability to the correct class. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- The loss becomes large when the model is highly confident and wrong. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- This makes log loss more appropriate than square loss for binary probability estimation. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Decision threshold
- To convert predicted probabilities into class labels, a threshold is used, often 0.5. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- If \(\hat{y}_i \ge 0.5\), predict the positive class; otherwise predict the negative class. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Changing the threshold changes precision and recall tradeoffs without changing the fitted model weights. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Practical rule
- Logistic Regression is widely used for text classification because it performs well with sparse, high-dimensional feature vectors. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/text-classification-using-logistic-regression/)

***

## Using Logistic Regression for Bounded Regression

### Core idea
- Because logistic outputs lie in \([0,1]\), the model can sometimes be used for targets that are naturally bounded in that interval. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- A target variable can also be rescaled into \([0,1]\) using min-max normalization and then mapped back later. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Min-max normalization
\[
\frac{x-\min}{\max-\min}
\]

### Interpretation
- This approach is only sensible when the target truly has meaningful lower and upper bounds. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- It is not a general replacement for standard regression methods. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Caution
- If the original problem is an ordinary continuous regression problem, OLS or another regression model is usually more natural. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Any transformation applied during training must be reversed correctly during evaluation and deployment. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

***

## Logistic Regression for Sentiment Analysis

### Standard workflow
1. Split the labeled dataset into training and evaluation subsets. [drlee](https://drlee.io/text-preprocessing-and-classification-with-logistic-regression-ea4fe3cfcaac)
2. Convert documents into bag-of-words or tf-idf vectors. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
3. Fit a Logistic Regression model on the training features. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/text-classification-using-logistic-regression/)
4. Predict on held-out data. [drlee](https://drlee.io/text-preprocessing-and-classification-with-logistic-regression-ea4fe3cfcaac)
5. Evaluate using metrics such as accuracy, precision, recall, and F1. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Why it works well
- Sentiment analysis often has enough lexical signal for a linear boundary in bag-of-words space to work surprisingly well. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/text-classification-using-logistic-regression/)
- Sparse linear models scale well to large vocabularies and many documents. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

### Important implementation rule
- Feature extraction must be fit on training data only to avoid leakage. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The model should be evaluated on held-out data rather than the same data used for fitting. [drlee](https://drlee.io/text-preprocessing-and-classification-with-logistic-regression-ea4fe3cfcaac)

### Strong baseline point
- Logistic Regression is often treated as a strong baseline for text classification because it is simple, fast, interpretable, and effective with sparse features. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/text-classification-using-logistic-regression/)

***

## Likely Test Targets

### High priority
- Know the general linear prediction formula and what each term means. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Distinguish **OLS** from **Logistic Regression** by target type, output type, and loss function. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Know why OLS is not ideal for binary class probabilities. [kb.osu](https://kb.osu.edu/bitstreams/5c7cd0a0-48b1-5fa3-9e62-b0c9b658105a/download)
- Know what the sigmoid function does. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Know what log loss penalizes and why it fits classification better than square loss. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Know that Logistic Regression is commonly used for sparse text classification. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

### Medium priority
- Understand the role of the intercept or bias. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Understand that changing the threshold changes predicted labels and evaluation metrics, but not the fitted model itself. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Understand why sparse text inputs make linear methods computationally attractive. [web.stanford](https://web.stanford.edu/~jurafsky/slp3/4.pdf)

### Likely quiz traps
- Thinking Logistic Regression is a regression model just because of its name. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Forgetting that OLS outputs are unbounded. [kb.osu](https://kb.osu.edu/bitstreams/5c7cd0a0-48b1-5fa3-9e62-b0c9b658105a/download)
- Confusing the linear score with the sigmoid-transformed probability. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Assuming thresholding is part of training rather than part of converting probabilities to labels. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

***

## General Rules

### Linear models
- Linear models combine features additively through learned weights. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- They are interpretable because each feature has a direct directional contribution to the score. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Regression vs classification
- Use regression models when the target is continuous. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Use classification models when the target is categorical. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- The same feature space can support either task, but the output interpretation and loss must match the problem. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### Loss functions
- The loss function defines what the model is optimized to do. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- OLS minimizes square loss. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- Logistic Regression minimizes log loss for binary targets. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Probabilities and thresholds
- Logistic Regression outputs probabilities or probability-like scores for the positive class. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)
- A threshold converts those probabilities into hard labels. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Different thresholds produce different precision-recall tradeoffs. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Sparse text features
- Text feature spaces are usually high-dimensional and sparse. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
- Linear models are attractive in this setting because they train and predict efficiently on sparse input. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

### Practical modeling
- Split train and test data before fitting vectorizers or models. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Learn feature extraction parameters on training data only. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Evaluate on held-out data to estimate generalization rather than memorization. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Limitations
- Linear models assume additive feature effects unless interactions are explicitly encoded. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- They may miss subtle linguistic structure, compositional meaning, and long-range dependencies. [web.stanford](https://web.stanford.edu/~jurafsky/slp3/4.pdf)

***

## Glossary

### **Bias / Intercept**
- A constant added to the weighted sum of features that shifts the prediction baseline. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### **Linear Prediction**
- The weighted sum of input features and learned coefficients, optionally plus a bias term. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### **Linear Classifier**
- A classifier based on a linear score in feature space, often combined with a threshold or probability transformation. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### **Ordinary Least Squares (OLS)**
- A linear regression method that minimizes the sum of squared errors. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### **Square Loss**
- A loss based on the squared difference between actual and predicted numeric values. [coursera](https://www.coursera.org/articles/linear-regression-vs-logistic-regression)

### **Logistic Regression**
- A linear model for classification that applies a sigmoid transformation to a linear score to estimate probabilities for binary outcomes. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### **Sigmoid Function**
- A function that maps any real-valued score into the interval \((0,1)\). [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### **Log Loss**
- A classification loss that strongly penalizes confident wrong probability predictions. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### **Threshold**
- A cutoff used to convert predicted probabilities into discrete class labels. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### **Sparse Matrix**
- A matrix storing mostly non-zero entries efficiently, which is essential for large text feature spaces. [web.stanford](https://web.stanford.edu/~jurafsky/slp3/4.pdf)

***

## What I improved

The biggest upgrades from your draft are:
- making the **OLS vs Logistic Regression** contrast much sharper and more testable. [kb.osu](https://kb.osu.edu/bitstreams/5c7cd0a0-48b1-5fa3-9e62-b0c9b658105a/download)
- explicitly separating **linear score**, **sigmoid probability**, and **thresholded class label**, which is a common confusion point. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- emphasizing why Logistic Regression is a strong fit for **bag-of-words and tf-idf text classification**. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/text-classification-using-logistic-regression/)

If you want, the next best step is for me to produce a **combined Week 2 study guide** from these three rewritten sections, with:
- **Executive Summary**
- **Core Concepts**
- **Likely Test Targets**
- **High/Medium/Low priority tags**
- and a **professor-profile calibration note** based on Quiz 1. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

---

Here’s a stronger, cleaner rewrite of **Code Material: Zipf’s Law TF-IDF Logistic Regression**. The main fixes are: it now reads more like a **study guide tied to implementation behavior**, it flags a few **important code-level concepts the quiz could target**, and it better separates **corpus statistics**, **feature construction**, **model fitting**, and **evaluation**. I also sharpened places where your original draft was conceptually right but too broad, especially around **DF vs TF**, **training-only fitting**, **sparse matrices**, and the distinction between `predict` and `predict_proba`. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

## Code Material: Zipf’s Law TF-IDF Logistic Regression

* **Summary:** This notebook walks through a full classical NLP pipeline for sentiment classification using the IMDB review dataset: preprocess text, inspect corpus frequency structure through Zipf’s law, build a vocabulary, convert documents into TF-IDF vectors, train Logistic Regression, and evaluate on held-out data. The core lesson is that **feature construction and data-splitting choices are as important as the classifier itself**. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

* **Why this notebook matters:** The lecture materials explain bag-of-words, TF-IDF, and linear classifiers conceptually, while this notebook shows how those ideas are implemented step by step. That makes it especially important for quiz preparation, because this instructor may ask about **what each code block is doing, why it is written that way, and what could go wrong if the steps are done in the wrong order**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)

* **Likely testing style:** Expect code-adjacent questions about **document frequency vs term frequency**, **why `set(tokens)` is used for DF**, **why sparse matrices are necessary**, **what `fit` learns versus what `transform` applies**, **what cross-validation is doing**, and **how `predict` differs from `predict_proba`**. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Overview

### Pipeline overview
- The notebook demonstrates a full text classification workflow:
  1. Load labeled data.
  2. Preprocess text.
  3. Analyze corpus frequency behavior.
  4. Build vocabulary and IDF statistics.
  5. Convert documents into TF-IDF features.
  6. Split data.
  7. Train Logistic Regression.
  8. Evaluate on held-out data.
  9. Run inference on new text. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

### Main theme
- The pipeline is not just “put text into a classifier.” [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Every earlier choice, including preprocessing, vocabulary pruning, and weighting, changes what information the model can learn from. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

***

## The Text Preprocessing Pipeline

### Core idea
- Raw text must be transformed into standardized tokens before machine learning models can use it effectively. [ixopay](https://www.ixopay.com/blog/what-is-nlp-natural-language-processing-tokenization)
- Common preprocessing operations include tokenization, lowercasing, stopword handling, and stemming or lemmatization. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Why preprocessing matters
- Preprocessing changes the vocabulary, and vocabulary changes the feature matrix. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- That means preprocessing is not just cleaning; it is part of model design. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

### Important rules
- The order of operations matters: tokenization must happen before token-level filtering or stemming. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Lowercasing before stopword filtering improves consistency because membership checks are case-sensitive. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- The same preprocessing pipeline must be used at both training time and inference time. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Likely testable point
- Stopword removal is task-dependent rather than universally correct. In sentiment tasks, some common-looking words may still carry useful signal. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

***

## Zipf’s Law and the Long Tail

### Core idea
- Natural language has a highly skewed frequency distribution: a few words are extremely common, while many words are rare. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- This is commonly described by **Zipf’s law**, where frequency decreases roughly as rank increases. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### Coding interpretation
- If you count corpus-wide token frequencies and sort terms by frequency, you should see a short high-frequency head and a long low-frequency tail. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- A log-log plot is useful because the values span multiple orders of magnitude. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### Why this matters for modeling
- Very frequent words are often weak discriminators. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- Very rare words may be noisy, corpus-specific, or too sparse to help much. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- This is why vocabulary pruning and weighting schemes are so important. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Likely testable point
- Zipf analysis produces **corpus statistics**, not document features. That distinction matters. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

***

## Feature Extraction: Bag-of-Words and TF-IDF

### Core idea
- Classifiers need numeric vectors, so each document must be mapped into a feature space defined by a vocabulary. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Common weighting schemes include binary features, term frequency, and TF-IDF. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Important distinction
- **TF** measures how often a term appears in one document. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- **DF** measures how many documents contain the term. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- **IDF** downweights terms with high document frequency. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Why TF-IDF is useful
- TF highlights terms that matter within a document. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- IDF reduces the influence of terms that appear in many documents across the corpus. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- TF-IDF combines both ideas and is often a strong baseline for text classification. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

### Likely testable point
- Feature extraction is a modeling step, not just a formatting step. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

***

## Sparse Matrices

### Why sparsity appears
- Text vocabularies can contain thousands or tens of thousands of terms, but any one document contains only a small fraction of them. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- That means most entries in the document-term matrix are zero. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Why sparse storage matters
- Sparse formats store mainly non-zero values and their positions instead of every zero explicitly. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- This saves large amounts of memory and speeds up training for text models. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)

### Practical rule
- Sparse matrices are the standard representation for bag-of-words and TF-IDF pipelines. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
- Many scikit-learn text tools already return sparse matrices by default. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Logistic Regression for Text Classification

### Core idea
- Logistic Regression is a linear classifier that converts a weighted feature sum into a probability-like output. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- It is widely used for binary text classification because it works well with sparse, high-dimensional features. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Interpretation of weights
- Positive weights push predictions toward the positive class. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Negative weights push predictions toward the negative class. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- The intercept shifts the baseline prediction. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Why it fits this notebook
- IMDB sentiment analysis is a standard binary classification problem with sparse text inputs, which makes TF-IDF plus Logistic Regression a strong classical baseline. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

### Likely testable point
- Logistic Regression predicts probabilities first; class labels come from thresholding those probabilities. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

***

## Model Evaluation

### Core principle
- A classifier should be evaluated on held-out data rather than only on the data used for fitting. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Evaluation metrics include accuracy, precision, recall, F1, and the confusion matrix. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Why multiple metrics matter
- Accuracy is simple but can hide important errors. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Precision focuses on false positives. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Recall focuses on false negatives. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- F1 balances precision and recall. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Likely testable point
- The confusion matrix provides the raw counts underlying these metrics, so understanding it helps derive precision, recall, and F1. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Python Libraries

### pandas
- Used to load and manipulate tabular datasets such as `movie_data.csv`. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)
- A DataFrame stores the review text and sentiment labels in a structured form. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

### nltk
- Used for tokenization, stopword handling, and stemming in the custom preprocessing pipeline. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### collections.defaultdict
- Useful for counting terms and document frequencies without manually checking whether keys already exist. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### re
- Used for regex-based cleaning, such as replacing punctuation with spaces before whitespace tokenization. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### matplotlib
- Used for Zipf plots and other exploratory visualizations. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### numpy
- Used for numerical operations and summaries in machine learning workflows. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### tqdm
- Used for progress bars in loops over many documents. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

### math
- Used for logarithms in manual TF and IDF calculations. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### sklearn
- Used for splitting data, vectorization, model fitting, cross-validation, and evaluation. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### scipy
- Used for sparse matrix structures such as CSR matrices. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### pickle
- Often used to save learned preprocessing objects or trained models for reuse. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

***

## Sentiment Analysis on the IMDB Dataset

### Dataset loading
```python
import pandas as pd

df = pd.read_csv('movie_data.csv')
print(len(df))
```

```text
50000
```

### Interpretation
- The notebook uses a dataset of 50,000 movie reviews with sentiment labels. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)
- This is a supervised text classification setup: raw text is the input and sentiment is the target label. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

### General rules
- Text datasets typically contain raw text plus labels, but the labels are not vectorized the way the text is. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The text must be transformed into features before fitting a classifier. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## NLTK-Based Preprocessing

### Stopwords and tokenization
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
```

### Case sensitivity example
```python
print('An' in stop)
print('an' in stop)
print("'s" in stop)
```

### Why this matters
- Membership tests in Python are case-sensitive, so lowercasing affects stopword filtering behavior. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Whether token fragments like `"'s"` are removed depends on the tokenizer output and the stopword list itself. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

### Preprocessing function
```python
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def pre_processing_by_nltk(doc, stemming=True, need_sent=False):
    sentences = sent_tokenize(doc)
    tokens = []
    for sent in sentences:
        words = word_tokenize(sent)
        if stemming:
            words = [ps.stem(word) for word in words]
        if need_sent:
            tokens.append(words)
        else:
            tokens += words
    return [w.lower() for w in tokens if w.lower() not in stop]
```

### General rules
- This pipeline performs sentence splitting, word tokenization, optional stemming, lowercasing, and stopword removal in sequence. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- The order matters because filtering and normalization operate on the produced tokens. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Consistency matters: the same function should be used whenever the model expects the same feature space. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Likely testable point
- A preprocessing function changes both vocabulary size and feature meaning, so changing it changes the model even if the classifier is unchanged. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## Testing Zipf’s Law

### Counting frequencies
```python
from collections import defaultdict
freq = defaultdict(int)
import re

corpus = ' '.join(list(df.review))
new_corpus = re.sub(r'[^\w\s]', ' ', corpus)

raw_tokens = new_corpus.lower().split()
for token in raw_tokens:
    freq[token] += 1
```

### Interpretation
- This code builds a corpus-level token frequency dictionary after regex cleaning and lowercasing. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- Because it uses whitespace splitting after punctuation removal, the resulting counts depend on this specific tokenization strategy. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### General rules
- Corpus statistics depend on preprocessing choices. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)
- Different cleaning rules produce different token counts and different Zipf plots. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### Frequency ranking
```python
order_tokens = sorted(list(freq.items()), key=lambda x: -x [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3))
```

### Important note
- To rank by descending frequency, the sort key should use the frequency value, not the full tuple. This corrected form better matches the intended interpretation of the notebook section. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### Visualization
```python
import matplotlib.pyplot as plt

y = [freq for token, freq in order_tokens]
plt.loglog(y)
```

### General rules
- A log-log plot is appropriate when frequencies span several orders of magnitude. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- Linear-scale plots often hide the tail because a few common terms dominate the range. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

***

## Building TF-IDF Ourselves

### Document frequency
```python
from tqdm import tqdm
DF = defaultdict(float)
for doc in tqdm(df.review):
    tokens = pre_processing_by_nltk(doc)
    for token in set(tokens):
        DF[token] += 1
```

### Why `set(tokens)` matters
- **Document frequency** counts whether a term appears in a document, not how many times it appears in that document. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- Using `set(tokens)` ensures each document contributes at most one count per term. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- Without the set conversion, the code would accidentally count term frequency instead of document frequency. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

### Vocabulary and IDF construction
```python
from math import log
IDF, vocab = dict(), dict()
for token in DF:
    if DF[token] < 50:
        pass
    else:
        vocab[token] = len(vocab)
        IDF[token] = log(1 + len(df.review) / DF[token])
```

### Interpretation
- This code prunes rare terms by requiring a minimum document frequency threshold. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- It then assigns each retained term a vocabulary index and an IDF value. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

### General rules
- Frequency thresholds control vocabulary size and reduce extreme sparsity. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- The exact IDF formula may vary, but the purpose remains the same: downweight terms that appear in many documents. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

### Handling unknown words
```python
IDF['<UNK>'] = 1
vocab['<UNK>'] = len(vocab)
```

### General rules
- An unknown token provides a fallback for terms outside the retained vocabulary. [ixopay](https://www.ixopay.com/blog/what-is-nlp-natural-language-processing-tokenization)
- This is one way to limit vocabulary growth and make feature extraction more robust to rare or unseen words. [ixopay](https://www.ixopay.com/blog/what-is-nlp-natural-language-processing-tokenization)

### TF-IDF feature function
```python
def tfidf_feature_extractor(doc, vocab, IDF):
    tokens = pre_processing_by_nltk(doc)
    for i, token in enumerate(tokens):
        if token not in vocab:
            tokens[i] = '<UNK>'
    TF = defaultdict(int)
    for token in tokens:
        TF[token] += 1
    x = [0] * len(vocab)
    for token in set(tokens):
        tfidf = log(TF[token] + 1) * IDF[token]
        token_id = vocab[token]
        x[token_id] = tfidf
    return x
```

### Why this is useful
- This hand-built version makes the logic of TF-IDF explicit: count term frequency, map unknowns, apply log-scaled TF, multiply by IDF, and place the result into the feature vector. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- It is educational because it exposes exactly what library tools automate. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Important limitation
- A dense Python list is conceptually simple but inefficient for large sparse vocabularies. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Library vectorizers are usually preferred in practice for efficiency and consistency. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

***

## Train/Test Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size=0.2, shuffle=True
)
```

### General rules
- Train/test splitting estimates generalization to unseen data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The test set should remain untouched during model fitting and model selection. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Shuffling is helpful when the dataset does not have meaningful order that should be preserved. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Critical rule
- Vocabulary learning and IDF estimation should ideally be done on training data only. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- Fitting these on the full dataset before splitting would cause leakage. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Logistic Regression Training

### Small training subsets
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train[:1000], y_train[:1000])
clf.score(X_test, y_test)
```

### General rules
- Small subsets are useful for debugging and quick experiments. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)
- More training data often improves performance, although gains may taper off. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

### Convergence warnings
```python
clf = LogisticRegression(random_state=0).fit(X_train[:10000], y_train[:10000])
```

### General rules
- A convergence warning means optimization stopped before full convergence. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- This suggests that training settings may need adjustment, such as increasing `max_iter`. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- It does not automatically mean the model is useless, but it should not be ignored. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

***

## Sparse Matrix Training

### CSR conversion
```python
import scipy.sparse as sparse
sparse_X = sparse.csr_matrix(X_train)
```

### Why it helps
- CSR stores only non-zero entries and their positions. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- This is well suited to document-term matrices, where most entries are zero. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### General rules
- Sparse matrices are usually preferable to dense matrices for large text datasets. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
- The savings can be substantial in both memory usage and training speed. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## CountVectorizer Example

```python
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = [
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'
]
bag = count.fit_transform(docs)
```

### What this demonstrates
- `CountVectorizer` learns a vocabulary during `fit` and converts documents into count vectors. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The output is a sparse document-term matrix. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Token inclusion depends on tokenization, case handling, and other preprocessing rules. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Likely testable point
- `fit_transform` both learns and applies the mapping, while `transform` only applies an already learned mapping. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## TF-IDF with Library Tools

### TfidfTransformer
```python
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
```

### General rules
- `TfidfTransformer` takes count vectors and reweights them into TF-IDF vectors. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- `smooth_idf=True` adjusts the IDF calculation to avoid edge-case issues. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- `norm='l2'` scales each document vector to unit L2 length, which helps reduce pure document-length effects. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

### TfidfVectorizer
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=pre_processing_by_nltk,
    use_idf=True,
    norm='l2',
    smooth_idf=True
)
```

### General rules
- `TfidfVectorizer` combines tokenization, vocabulary learning, counting, and TF-IDF weighting in one object. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- It is conceptually similar to `CountVectorizer` followed by `TfidfTransformer`. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Fitting learns the vocabulary and IDF statistics from training data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Transforming new text reuses the learned mapping without rebuilding the vocabulary. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Cross-Validated Logistic Regression

```python
from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(
    cv=5,
    scoring='accuracy',
    random_state=0,
    n_jobs=-1,
    verbose=3,
    max_iter=300
).fit(X_train, y_train)
```

### What it does
- `LogisticRegressionCV` performs internal cross-validation to choose regularization settings automatically. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Cross-validation gives a more robust estimate of training-time model quality than a single split alone. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### General rules
- The chosen scoring metric affects which model configuration is selected. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Internal cross-validation should still remain separate from final test evaluation. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

***

## Predictions and Probabilities

```python
yhat = clf.predict(X_test)
clf.predict_proba(x_ins)
```

### Important distinction
- `predict` returns class labels. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- `predict_proba` returns estimated probabilities for each class. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Probability outputs are useful for thresholding, ranking, and confidence-aware decisions. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Likely testable point
- Probabilities and labels are not the same object; labels are derived from probabilities using a decision rule. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

***

## Confusion Matrix and Metrics

```python
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, yhat)
```

### Manual metrics
```python
prec = confusion_mat / (confusion_mat + confusion_mat) [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
rec = confusion_mat / (confusion_mat + confusion_mat) [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
f1 = 2 * prec * rec / (prec + rec)
```

### Important note
- The displayed manual metric example appears conceptually intended to illustrate deriving precision, recall, and F1 from confusion counts, but library metric functions are safer in practice because hand-written formulas are easy to get wrong. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The important learning goal is understanding what the confusion matrix counts mean and how those counts support metric computation. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### General rules
- Precision measures the fraction of predicted positives that are correct. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Recall measures the fraction of actual positives that are recovered. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- F1 combines precision and recall into one score. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Example Inference

```python
text = 'this is my favorite movie!'
x_ins = tfidf.transform([text])
clf.predict(x_ins)
clf.predict_proba(x_ins)
```

### General rules
- New text must be transformed with the same fitted vectorizer used during training. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- The new document must live in the same feature space as the training data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Terms unseen during fitting are ignored or handled according to the vectorizer’s vocabulary behavior. [ixopay](https://www.ixopay.com/blog/what-is-nlp-natural-language-processing-tokenization)

### Practical meaning
- A deployed text classifier is really a pipeline: preprocessing plus vectorization plus model prediction. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)

***

## Global Rules from This Notebook

### Pipeline rules
- A complete text classification workflow includes loading data, preprocessing, feature construction, splitting, training, evaluation, and inference. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)
- Skipping or misordering steps can change results substantially. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Representation rules
- Vocabulary design, term filtering, and weighting strongly affect model behavior. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- TF-IDF often improves over raw counts when common words would otherwise dominate the feature space. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)
- Sparse representations are standard for high-dimensional text data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Modeling rules
- Logistic Regression is a strong baseline for binary text classification. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
- More training data often helps, but optimization and representation choices matter too. [kaggle](https://www.kaggle.com/code/harshildarji/imdb-sentiment-classification-using-tf-idf)
- Convergence warnings indicate optimization issues, not automatic model failure. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Evaluation rules
- Test data must remain separate from training and tuning. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Confusion-matrix-based metrics give more detail than accuracy alone. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Probability outputs support thresholding and decision analysis. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

***

## Likely Test Targets

### High priority
- Explain why `set(tokens)` is used when computing DF. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- Distinguish TF, DF, IDF, and TF-IDF. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- Explain why sparse matrices are necessary in NLP feature extraction. [scikit-learn](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
- Know what `CountVectorizer`, `TfidfTransformer`, and `TfidfVectorizer` each do. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Know the difference between `predict` and `predict_proba`. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Know why vectorizers must be fit on training data only. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### Medium priority
- Explain the modeling purpose of `<UNK>`. [ixopay](https://www.ixopay.com/blog/what-is-nlp-natural-language-processing-tokenization)
- Explain why `smooth_idf=True` and L2 normalization are useful. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Explain what a convergence warning means in Logistic Regression. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Likely quiz traps
- Confusing corpus-level frequency analysis with document-level features. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- Confusing TF with DF. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)
- Treating `fit_transform` and `transform` as the same operation. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- Forgetting that evaluation on the training set does not measure generalization. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

***

## Glossary

### **CSR Matrix**
- A compressed sparse row matrix format that stores mainly non-zero entries efficiently for sparse text data. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Document Frequency (DF)**
- The number of documents that contain a term at least once. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### **TF**
- A document-level quantity measuring how often a term appears within one document. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

### **IDF**
- A corpus-level weighting factor that reduces the influence of terms appearing in many documents. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

### **TF-IDF**
- A weighting scheme combining within-document frequency and across-corpus rarity. [developers.google](https://developers.google.com/machine-learning/guides/text-classification/step-3)

### **TfidfTransformer**
- A tool that converts count-based document-term features into TF-IDF-weighted features. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **TfidfVectorizer**
- A tool that directly converts raw text documents into TF-IDF feature matrices. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **CountVectorizer**
- A tool that converts raw text into sparse count vectors over a learned vocabulary. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)

### **Unknown Token (`<UNK>`)**
- A fallback symbol used to represent terms outside the retained vocabulary. [ixopay](https://www.ixopay.com/blog/what-is-nlp-natural-language-processing-tokenization)

### **Zipf’s Law**
- The empirical pattern that word frequency decreases roughly as rank increases, creating a high-frequency head and a long tail. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/423bdeda-10f4-4901-b543-c5b4988d2271/week01_content.md)

***

## What I improved

The biggest upgrades from your version are:
- making the notebook read as a **pipeline with testable implementation logic**, not just a long walkthrough. [scikit-learn](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- explicitly flagging **DF vs TF**, **training-only fitting**, **sparse representation**, and **predict vs predict_proba**, which are prime quiz material. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
- tightening some code explanations so they reflect the intended conceptual lesson more clearly, especially for frequency ranking and manual metric calculation. [afit-r.github](https://afit-r.github.io/tf-idf_analysis)

If you want, I can now turn all four Week 2 sections into one **final polished Week 2 study guide** with:
- **Executive Summary**
- **Core Concepts**
- **Likely Test Targets**
- **High / Medium / Low priority**
- and a **“what changed from your original notes”** section. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/e92a01c3-bd3b-4c29-860f-29ebe6424267/quiz01.md)
