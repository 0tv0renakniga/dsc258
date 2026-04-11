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

# Lecture Material: Bag of Words Classifier

* **Summary:**This lecture introduces a classical text classification workflow based on bag-of-words representations, linear classifiers, and standard evaluation methods. It also adds generalized rules about how document-term features are built, how data should be split to avoid leakage, and how to interpret common classification metrics.

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
- [Generalized Rules](#generalized-rules)
- [Glossary](#glossary)

## Course and Lecture Context

### Lecture Title
- **Bag of Words Classifier**

### Summary
- Outlines a general framework for text mining and natural language processing tasks.
- Focuses on representation learning as a crucial step.
- Introduces classical bag-of-words representation for document classification.
- Connects representation choices to model evaluation and validation strategy. 

### Goals
- Describe the general framework for text mining and natural language processing tasks.
- Apply bag-of-words representations to conduct text classification.
- Describe and apply the implementation details of bag-of-words representation.
- Explain how vocabulary construction, feature weighting, and evaluation choices affect classifier behavior. 

## Bag of Words Classifier

### Main Topic
- The lecture introduces the **Bag-of-Words Text Classifier**.

### Purpose
- The focus is on defining text classification and understanding how to evaluate a classifier.
- It covers foundational **Bag-of-Words Representations**.
- It introduces building a linear classifier upon these bag-of-words representations.
- It emphasizes that representation design is a central part of the modeling pipeline, not just a preprocessing detail. 

### Core Idea
- A bag-of-words classifier converts each document into a numeric feature vector over a vocabulary and then uses those features as input to a classifier. 
- The representation typically keeps track of which terms occur and how often they occur, while discarding most grammar and long-range word order. 

## A Typical Text Mining Pipeline

### Pipeline Overview
- The lecture recaps the following sequence:

$$
\text{Preprocessing} \rightarrow \text{Representation} \rightarrow \text{Model Training}
$$

### High-Level Meaning
- The pipeline is organized into three major stages:
  - **Preprocessing**: Try to make texts "better formatted".
  - **Representation**: Make the text data machine-actionable.
  - **Model Training**: Specifically for the end-task.

### Interpretation
- In text classification, preprocessing prepares raw text, representation converts it to features such as counts or tf-idf weights, and model training fits a predictive function on those features. 
- The quality of the representation strongly affects downstream performance because the classifier only sees the numeric features, not the raw text itself. 

## Sentiment Analysis as an Example

### Main Topic
- Using **Sentiment Analysis** to demonstrate text classification.

### Purpose
- To estimate a document's sentiment from its content, such as determining whether a review is positive or negative.
- Utilizes datasets like the IMDB movie reviews dataset to classify textual data into positive (1) or negative (0) sentiments.

### Why This Example Fits
- Sentiment analysis is a standard document classification task in which the input is free text and the output is a discrete label. 
- It is a good match for bag-of-words methods because sentiment is often signaled by word usage patterns that can be captured by document-term features. 

## Bag-of-Words Representation

### Definition
- **Bag-of-words** represents a document as a fixed-length vector over a learned vocabulary. 
- Each dimension corresponds to a term, and the value usually reflects a count, a binary indicator, or a weighted value such as tf-idf. 

### What It Keeps
- Whether a term appears in the document.
- How often a term appears, if counts are used.
- Limited local order information when n-grams are included. 

### What It Discards
- Most word order.
- Most syntactic structure.
- Most long-range contextual meaning. 

### Consequence
- Two documents with similar term counts can receive similar vectors even if their phrasing or exact meaning differs. 
- This makes bag-of-words simple and effective for many tasks, but it also limits its ability to model nuanced semantics. 

## Vectorization Rules

### CountVectorizer
- `CountVectorizer` converts a collection of text documents into a matrix of token counts. 
- The output is typically a **sparse matrix**, because most documents contain only a small fraction of the full vocabulary. 

### Default Behavior
- By default, `CountVectorizer` lowercases text before tokenization. 
- By default, it uses a token pattern that selects tokens of 2 or more alphanumeric characters. 
- Punctuation generally acts as a separator rather than a feature token under the default tokenization pattern. 

### Vocabulary Learning
- The vocabulary is learned during `fit` from the training corpus. 
- When transforming new documents, terms that were not seen during training are ignored. 
- Because of this, train/test or train/validation splits must be created before fitting the vectorizer to avoid leakage. 

### Frequency Controls
- `min_df` removes terms that are too rare across the corpus. 
- `max_df` removes terms that appear in too many documents and can function like corpus-specific stop-word filtering. 
- `binary=True` changes features from counts to presence/absence indicators. 

## N-grams and TF-IDF

### N-grams
- A **unigram** is a single token.
- A **bigram** is a sequence of two adjacent tokens.
- Higher-order n-grams capture more local word order than plain unigram bag-of-words. 

### Why N-grams Matter
- Unigrams ignore order almost completely.
- Bigrams and higher n-grams can preserve short local patterns such as `not good`, which may be important for classification. 

### TF-IDF
- TF-IDF stands for **term frequency-inverse document frequency**. 
- It reweights count-based features so that terms frequent in a document but not too common across the full corpus receive higher importance. 
- `TfidfVectorizer` is equivalent in spirit to applying `CountVectorizer` followed by TF-IDF reweighting. 

### Interpretation
- Raw counts can overemphasize very common words.
- TF-IDF often improves document classification by downweighting common corpus-wide terms and highlighting more discriminative terms. 

## Data Validation and Splitting

### K-fold Cross Validation
- A method of evaluating a model by splitting the available training data into $k$ folds and repeating training/evaluation $k$ times. 
- In each iteration, one fold is held out for evaluation and the remaining $k-1$ folds are used for training. 
- The evaluation scores are then aggregated across folds to estimate model performance more robustly. 

### Important Clarification
- In standard practice, cross-validation is typically performed on the training data for model selection or hyperparameter tuning. 
- A separate **final test set** should still be held out for one-time final evaluation. 

### Train/Dev/Test
- The standard practice of partitioning a dataset into three distinct segments:
  - **Train**: For fitting the model.
  - **Validation (Dev)**: For tuning parameters and making adjustments.
  - **Test**: For the final evaluation of the model.

### Leakage Rule
- The vectorizer and any learned preprocessing statistics must be fit only on the training split within each experiment. 
- Validation and test data should only be transformed using objects fit on training data, never used to build the vocabulary directly. 

## Evaluation Metrics

### Predicted vs. Actual Values
- The foundation of evaluating classification models relies on comparing Predicted Values against Actual Values:
  - **TP (True Positive)**: Predicted as Positive and true label is Positive.
  - **TN (True Negative)**: Predicted as Negative and true label is Negative.
  - **FP (False Positive)**: Predicted as Positive and true label is Negative.
  - **FN (False Negative)**: Predicted as Negative and true label is Positive.

### Accuracy, Precision, Recall, and F1

- **Accuracy**
  - The proportion of total correct predictions.
  $$\text{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN}$$

- **Precision**
  - The proportion of predicted positives that are actually positive.
  $$\text{Precision}=\frac{TP}{TP+FP}$$

- **Recall**
  - The proportion of actual positives that are identified correctly.
  $$\text{Recall}=\frac{TP}{TP+FN}$$

- **F1 Score**
  - The harmonic mean of Precision and Recall.
  $$F1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

### Interpretation Rules
- **Accuracy** is easy to understand but can be misleading when classes are imbalanced, because a classifier can perform well on the majority class while missing many minority examples. 
- **Precision** is especially important when false positives are costly. 
- **Recall** is especially important when false negatives are costly. 
- **F1** is useful when both precision and recall matter and a single combined score is needed. 

### Multiclass Note
- In multiclass classification, precision, recall, and F1 are often aggregated using **micro**, **macro**, or **weighted** averaging. 
- These averaging methods summarize per-class behavior in different ways and may produce different conclusions when classes are imbalanced. 

## Generalized Rules

### Representation Rules
- Bag-of-words is a family of document representations based on vocabulary features, not just a single exact formula. 
- Features may be raw counts, binary indicators, or tf-idf weights. 
- N-grams extend bag-of-words by capturing short local sequences instead of only single tokens. 

### Modeling Rules
- A classifier trained on bag-of-words features learns correlations between vocabulary patterns and labels, not deep semantic understanding. 
- Linear classifiers are commonly paired with bag-of-words because sparse text features work well with linear decision boundaries in many practical settings. 

### Vocabulary Rules
- The learned vocabulary depends on preprocessing choices, tokenization rules, and frequency thresholds. 
- Unseen words at inference time do not create new columns; they are ignored unless the vocabulary is rebuilt. 

### Evaluation Rules
- Model quality depends not only on the classifier, but also on feature design, data splitting, and metric choice. 
- No single metric is always best; the right metric depends on the task and on the relative costs of false positives and false negatives. 

### Practical Rules
- Always split data before learning vocabulary or feature weights. 
- Use cross-validation for model selection when data is limited, but keep a final test set for final reporting when possible. 
- Prefer tf-idf or filtered vocabularies when very common words dominate count-based features. 

## Glossary

### **Accuracy**
- An evaluation metric representing the proportion of correct predictions out of all predictions.

### **Bag-of-words**
- A document representation model that maps text to a vocabulary-based vector, usually ignoring most word order while preserving token occurrence information. 

### **Binary Features**
- A feature setting in which each term records only whether it appears, rather than how many times it appears. 

### **CountVectorizer**
- A scikit-learn tool that converts a collection of text documents into a sparse matrix of token counts. 

### **Cross-validation**
- A model evaluation procedure that repeatedly splits training data into training and held-out folds to estimate performance more robustly. 

### **Document-Term Matrix**
- A matrix in which rows correspond to documents and columns correspond to vocabulary terms, with entries storing counts or weights. 

### **F1 Score**
- An evaluation metric that combines Precision and Recall using their harmonic mean.

### **Feature Extraction**
- The process of converting raw text into numeric representations suitable for machine learning models. 

### **Inverse Document Frequency (IDF)**
- A corpus-level weighting factor that reduces the influence of terms that appear in many documents. 

### **K-fold Cross Validation**
- A validation technique that partitions training data into $k$ folds, trains the model repeatedly, and evaluates on a different held-out fold in each iteration. 

### **N-gram**
- A contiguous sequence of $n$ tokens, such as a unigram or bigram, used as a text feature. 

### **Precision**
- An evaluation metric measuring how many predicted positive instances are actually positive.

### **Recall**
- An evaluation metric measuring how many actual positive instances are correctly identified.

### **Sentiment Analysis**
- A text classification task that predicts sentiment labels such as positive or negative from document content.

### **Sparse Matrix**
- A matrix representation optimized for data in which most entries are zero, which is common in text feature extraction. 

### **TF-IDF**
- A feature weighting scheme based on term frequency and inverse document frequency. 

### **TfidfVectorizer**
- A scikit-learn vectorizer that converts text documents directly into tf-idf-weighted feature vectors. 

### **Train/Dev/Test**
- The standard three-way split of data used to train, tune, and formally evaluate a model while reducing data leakage risk.

### **Vocabulary**
- The set of terms selected as features for a bag-of-words or tf-idf representation. 

---

# Lecture Material: Bag of Words Representations

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
- [General Rules](#general-rules)
- [Summary of Bag of Words](#summary-of-bag-of-words)
- [Glossary](#glossary)

## Course and Lecture Context

### Lecture Title
- **Bag of Words Representations**

### Purpose
- The lecture details the implementation of Bag-of-Words representations for text classification.
- It explores how to convert text into machine-actionable vectors by defining terms and assigning them mathematical weights.
- It emphasizes that representation is a modeling choice: different definitions of terms and different weighting schemes can produce very different feature spaces.

### Goals
- Explain the bag-of-words assumption.
- Describe how documents are converted into vectors over a vocabulary.
- Compare binary, TF, and TF-IDF weighting.
- Understand why normalization and document frequency matter.
- Recognize the strengths and limits of bag-of-words representations.

## Bag of Words Assumption

### Core Concept
- The fundamental assumption of this model is to **ignore word ordering but keep the terms and their weights**.
- In the simplest setting, the weights are based on whether a term appears or how often it appears.
- The representation treats a document as a collection, or “bag,” of terms rather than a sequence.

### Interpretation
- Word order, syntax, and long-range context are mostly discarded.
- Documents with similar term counts can receive similar representations even if their phrasing differs.
- This simplification often works well for classification, but it cannot capture all aspects of meaning.

### General Rule
- Bag-of-words is useful when the presence or frequency of terms matters more than exact grammar or sentence structure.
- Bag-of-words is less suitable when word order is essential to meaning, such as in cases of negation, compositional semantics, or syntax-heavy tasks.

## Terms and Vocabulary

### What Is a Term?
- A **term** is the basic unit used as a feature in the vector representation.
- Depending on preprocessing and modeling choices, a term may be:
  - A word
  - A stemmed word
  - A lemmatized word
  - A phrase
  - An n-gram
  - Another normalized textual unit

### Vocabulary
- The **vocabulary** is the full set of terms used as feature dimensions.
- If the vocabulary is $V=\{v_1,v_2,\dots,v_n\}$, then every document is represented as a vector of length $n$.

### General Rules
- The vocabulary depends on preprocessing choices such as tokenization, stemming, lemmatization, stop-word removal, phrase mining, and frequency filtering.
- Changing the vocabulary changes the feature space, which can change model performance.
- Terms not included in the vocabulary do not contribute to the final vector.

## Weighting Schemes

### How to Define Weights?
- Weights should reflect the importance of a term in a document.
- Common choices include:
  - **Binary**
  - **Term Frequency (TF)**
  - **TF-IDF**

### General Principle
- A good weighting scheme gives higher values to terms that are more informative for distinguishing documents.
- Different tasks may favor different weighting schemes:
  - Binary weights often work well when presence matters more than repetition.
  - Raw TF helps when repeated occurrence carries useful signal.
  - TF-IDF helps when common corpus-wide words should be downweighted.

### General Rule
- The weighting scheme is part of the model design, not just a formatting choice.
- There is no single universally best weighting scheme; performance depends on the task, corpus, and classifier.

## Binary Bag-of-Words

### Definition
- In binary bag-of-words, each feature records whether a term appears at least once in the document.
- Given a vocabulary $V=\{v_1,v_2,\dots,v_n\}$ and document $d=(w_1,w_2,\dots,w_m)$, the vector is:

$$
x_{d,i}=
\begin{cases}
1 & \exists j \text{ such that } v_i=w_j \
0 & \text{otherwise}
\end{cases}
$$

### Interpretation
- This representation ignores repeated occurrences after the first appearance.
- It captures **presence/absence**, not intensity.

### General Rules
- Binary weighting is often helpful when repeated use of a term adds little extra information.
- Binary weighting can be more robust than raw counts for short texts or highly repetitive texts.
- Binary weighting loses information about emphasis through repetition.

## Term Frequency (TF) Variants

### From Binary to TF
- Moving beyond binary, weights can be assigned based on **frequency**: how many times a term appears in a document.
- The count-based vector representation is:

$$
x_{d,i}=|\{j \mid w_j=v_i\}|
$$

### Core Idea
- A term is often considered more important if it occurs more frequently in a document.
- Let $f(t,d)$ be the raw frequency of term $t$ in document $d$.

### Popular Variants
- **Raw TF:** $TF(t,d)=f(t,d)$
- **Log TF:** $TF(t,d)=\log(f(t,d)+1)$
- **Maximum Frequency Normalization:** $TF(t,d)=0.5+0.5\times \frac{f(t,d)}{MaxFreq(d)}$
- **BM25-style TF normalization:** adjusts term frequency based on document length so long documents are not automatically favored

### Why Variants Exist
- Repeating a word 20 times is usually not 20 times more informative than using it once.
- Raw counts can overemphasize long or repetitive documents.
- TF normalization tries to preserve signal from repetition without letting verbosity dominate.

### General Rules
- Raw TF is simple and interpretable, but it can bias the model toward longer documents.
- Log TF reduces the effect of repeated occurrences by making additional repetitions contribute less than earlier ones.
- Length-normalized variants are useful when documents vary substantially in size.
- TF weighting assumes repeated occurrence provides some additional evidence, but usually with diminishing returns.

### Normalization
- Normalization is critical because repeated occurrences are generally less informative than the first occurrence.
- Good normalization reduces bias from document length while preserving meaningful differences in emphasis.

### General Rule
- If document lengths vary widely, normalized TF variants are usually safer than raw counts.
- If repetition is highly informative in the task, raw or lightly normalized TF may still work well.

## Inverse Document Frequency (IDF)

### Concept
- The core idea is that a term is more discriminative if it appears in fewer documents across the corpus.
- Let $df_D(t)$ be the number of documents in corpus $D$ that contain term $t$.

### Intuition
- A term that appears in nearly every document is usually less helpful for distinguishing among documents.
- A term that appears in only a subset of documents is often more informative.

### Variants of IDF
- **Log IDF:** $IDF(t)=1+\log\left(\frac{|D|}{df_D(t)}\right)$
- **Smoothed IDF:** $IDF(t)=\log\left(\frac{|D|+1}{df_D(t)+0.5}\right)$

### General Rules
- IDF downweights common terms and upweights rarer terms.
- Extremely common terms often receive low IDF because they provide little discriminative value.
- Extremely rare terms may receive very high IDF, but they are not always useful if they are noise, misspellings, or accidental one-offs.
- IDF is corpus-dependent: the same term can receive different weights in different collections.

## TF-IDF Weighting and Zipf's Law

### TF-IDF Weighting
- Combining both metrics yields:

$$
Weight(t,d)=TF(t,d)\times IDF(t)
$$

- A term receives a **high weight** when it is common in a specific document but rare in the overall collection.

### Interpretation
- TF captures **document-specific importance**.
- IDF captures **corpus-level discriminativeness**.
- TF-IDF combines both local and global information.

### General Rules
- TF-IDF is often a strong default for document classification and retrieval.
- TF-IDF is especially helpful when raw counts are dominated by frequent generic words.
- TF-IDF is still a bag-of-words model, so it does not solve the loss of word order or syntax.

### Zipf's Law (Revisit)
- Zipf's Law states that:

$$
\text{Rank} \times \text{Frequency} \approx \text{Constant}
$$

- A common functional form is:

$$
F(w)=\frac{C}{r(w)^{\alpha}}
$$

### Connection to Bag-of-Words
- **Stop words** occupy the highest-frequency region.
- The most useful content words often lie in the middle of the distribution.
- Very rare words form a long tail.

### General Rules
- High-frequency words are often too common to be discriminative.
- Very low-frequency words may be too sparse, noisy, or corpus-specific to be reliable.
- Mid-frequency words are often the most useful features in classical text mining.
- TF-IDF helps rebalance the influence of terms across the Zipfian frequency distribution.

## Sparse Representation

### Sparse Vector/Matrix
- In bag-of-words models, the vocabulary is usually very large, but each document contains only a small fraction of the terms.
- Therefore, most entries in a document vector are zero.

### Why Sparsity Matters
- Sparse vectors and sparse matrices store only non-zero values.
- This makes bag-of-words modeling computationally feasible even for large vocabularies.

### General Rules
- Sparse data structures are the standard representation for bag-of-words models.
- Efficient storage and computation become increasingly important as the vocabulary grows.
- Many classical linear models work especially well with sparse text features.

## General Rules

### Representation Design
- A bag-of-words model depends on two choices:
  - How to define the terms
  - How to assign the weights
- Both choices affect interpretability, sparsity, and downstream model performance.

### Preprocessing Dependence
- Bag-of-words quality depends strongly on preprocessing.
- Tokenization, stemming, lemmatization, stop-word removal, frequency filtering, and phrase construction all change the final vectors.

### Frequency Effects
- Very common words tend to be less useful for discrimination.
- Very rare words may be too sparse or noisy.
- Weighting and filtering are used to balance these extremes.

### Independence Assumption
- Bag-of-words effectively assumes that terms contribute independently once converted into vector features.
- This simplification is useful in practice, but it ignores syntax, compositional meaning, and interactions among distant words.

### Task Dependence
- No single term definition or weighting scheme is always best.
- Binary, TF, and TF-IDF should be understood as task-dependent design options rather than universally correct choices.

### Practical Rules
- Use binary features when presence matters more than repetition.
- Use TF when repetition carries useful signal.
- Use TF-IDF when common corpus-wide words should be downweighted.
- Use sparse representations for efficient storage and computation.
- Expect tradeoffs between simplicity, interpretability, and expressive power.

## Summary of Bag of Words

### Pros
- Empirically effective in practice.
- Intuitive to understand.
- Easy to implement.
- Works well with large vocabularies and sparse linear models.
- Often provides a strong baseline for document classification tasks.

### Cons
- Ignores syntax and most word order.
- Assumes term independence.
- Requires design choices for preprocessing, vocabulary definition, weighting, and filtering.
- Can struggle with negation, phrase-level meaning, and semantic similarity.
- Rare-word noise and high-frequency stop words must be handled carefully.

### Tips
- A **Sparse Vector/Matrix** is your best friend in vector space modeling.
- Mid-frequency terms are often more useful than extremely common or extremely rare ones.
- TF-IDF is often a strong baseline, but it is not automatically best for every dataset.
- Always treat vocabulary design and weighting as modeling decisions, not fixed defaults.

## Glossary

### **Bag-of-words Assumption**
- The modeling simplification where word ordering is ignored and a document is represented through its terms and their weights.

### **Binary Bag-of-Words**
- A bag-of-words weighting scheme in which each feature indicates only whether a term appears in a document.

### **BM25-style TF**
- A term-frequency variant that normalizes word counts by document length so longer documents do not dominate simply due to verbosity.

### **Inverse Document Frequency (IDF)**
- A corpus-level weighting factor that increases the importance of terms that occur in fewer documents.

### **Sparse Vector/Matrix**
- A data structure that stores only non-zero feature values, making large bag-of-words representations practical.

### **Term**
- A textual unit used as a feature in a bag-of-words model, such as a word, stem, lemma, phrase, or n-gram.

### **Term Frequency (TF)**
- A weighting scheme based on how many times a term appears within a document.

### **TF-IDF**
- A combined weighting scheme that highlights terms that are frequent in a document but relatively rare across the corpus.

### **Vocabulary**
- The complete set of terms used as dimensions in the vector representation.

### **Zipf's Law**
- An empirical pattern in language showing that a word’s frequency is inversely related to its rank, which helps explain why both very common and very rare words can be problematic in text mining.

---

# Lecture Material: Building Linear Classifiers

* **Summary:** This lecture explains how to move from text representations, such as bag-of-words vectors, to predictive models. It introduces linear prediction, Ordinary Least Squares for regression, Logistic Regression for classification, and practical rules for training and evaluating linear models on high-dimensional text data.

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [Building Linear Classifiers](#building-linear-classifiers)
- [Linear Prediction](#linear-prediction)
- [Ordinary Least Square (OLS) Regression](#ordinary-least-square-ols-regression)
- [From OLS to Logistic Regression](#from-ols-to-logistic-regression)
- [Logistic Regression for Classification](#logistic-regression-for-classification)
- [Using Logistic Regression for Bounded Regression](#using-logistic-regression-for-bounded-regression)
- [Logistic Regression for Sentiment Analysis](#logistic-regression-for-sentiment-analysis)
- [General Rules](#general-rules)
- [Glossary](#glossary)

## Course and Lecture Context

### Lecture Title
- **Building Linear Classifiers**

### Summary
- Extends the discussion of text classification and evaluation.
- Builds upon bag-of-words representations to implement machine learning models.
- Focuses specifically on building linear models, including Ordinary Least Squares and Logistic Regression, using these text features.
- Emphasizes the relationship between model choice, loss function, prediction type, and evaluation.

### Goals
- Understand how linear models map feature vectors to predictions.
- Distinguish regression from classification settings.
- Describe the prediction functions and loss functions used in OLS and Logistic Regression.
- Apply linear models to sparse, high-dimensional text features.
- Recognize the assumptions, strengths, and limitations of linear classifiers.

## Building Linear Classifiers

### Main Topic
- The transition from representing text data to training predictive models.

### Purpose
- To understand how to take a feature vector, such as a bag-of-words representation, and apply mathematical models to predict outcomes.
- These outcomes may be:
  - Continuous values, as in regression
  - Discrete categories, as in classification

### Core Idea
- A linear model computes a weighted combination of input features and optionally applies a transformation to that score.
- The model learns weights that make predictions align as closely as possible with observed labels.

### General Interpretation
- Features with larger positive weights push predictions upward or toward the positive class.
- Features with larger negative weights push predictions downward or toward the negative class.
- The intercept or bias term shifts the prediction even when all feature values are zero.

## Linear Prediction

### General Form
- A linear prediction function combines features and weights:

$$
\hat{y}_{i}=\sum_{j=1}^{d} x_{i,j} w_j + b
$$

- $x_i$ is the feature vector of the $i$-th instance.
- $d$ is the number of feature dimensions.
- $w_j$ is the weight for feature $j$.
- $b$ is the bias or intercept.
- $\hat{y}_i$ is the predicted score.

### Interpretation
- Each feature contributes proportionally to its value and weight.
- The prediction is additive: each feature pushes the score independently.
- In text classification, the input vector is often sparse, so only a small number of terms contribute for any given document.

### General Rules
- Linear models are especially effective when useful information is distributed across many sparse features.
- The sign and magnitude of a weight indicate how that feature influences the prediction.
- Linear prediction itself is not limited to classification or regression; the task is determined by the output interpretation and the loss function.

## Ordinary Least Square (OLS) Regression

### OLS: Example
- A classic example involves a dataset detailing the average heights and weights for American women aged 30–39.
- **Goal:** Build a mathematical relationship between Weight and Height:

$$
Weight = f(Height)
$$

### OLS: Linear Prediction
- OLS uses the linear prediction rule:

$$
\hat{y}_{i}=\sum_{j=1}^{d}x_{i,j}w_{j}+b
$$

### OLS: Square Loss Function
- **Square Loss for the $i^{th}$ instance:**

$$
l(y_i,\hat{y}_i)=(y_i-\hat{y}_i)^2
$$

- **Square loss for all $n$ instances:**

$$
L=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

### Interpretation
- OLS chooses parameters that minimize the sum of squared prediction errors.
- Squaring makes larger errors contribute disproportionately more than smaller ones.
- Because of this, OLS is sensitive to large errors and outliers.

### General Rules
- OLS is a regression model, not a probability model.
- OLS predicts unbounded real values, so it is appropriate when the target is continuous.
- Square loss strongly penalizes large deviations, which can be useful when large errors are especially undesirable.
- OLS assumes the target is numeric and that a linear relationship is a reasonable approximation.

### OLS via sklearn
- The standard OLS implementation minimizes square loss.
- In its basic form, it does not impose regularization.
- Parallelization parameters may affect training efficiency, but not the mathematical objective.

## From OLS to Logistic Regression

### Why OLS Is Not Enough for Classification
- In binary classification, labels are discrete, often 0 or 1.
- A plain linear predictor can output any real number, including values below 0 or above 1.
- This makes raw OLS outputs difficult to interpret as probabilities.

### Logistic Prediction
- Logistic Regression applies a sigmoid transformation to the linear score:

$$
\hat{y}_{i}=\sigma\left(\sum_{j=1}^{d}x_{i,j}w_j+b\right)
$$

- $\hat{y}_i$ is now interpreted as a probability-like quantity $\in [0,1]$

### Sigmoid Function
- The sigmoid function maps any real-valued input into the interval $(0,1)$:

$$
\sigma(x)=\frac{e^x}{e^x+1}=\frac{1}{1+e^{-x}}
$$

### Interpretation
- Large positive scores map to values near 1.
- Large negative scores map to values near 0.
- Scores near 0 map to probabilities near 0.5.

### General Rules
- Logistic Regression keeps a linear decision structure in feature space but uses a non-linear transformation on the output score.
- The sigmoid makes predictions interpretable as probabilities for binary outcomes.
- Logistic Regression is used for classification even though its name contains “regression.”

## Logistic Regression for Classification

### Target and Prediction
- In binary Logistic Regression:
  - The actual target satisfies $(y_i \in \{0,1\})$
  - The predicted output satisfies $\hat{y}_i \in [0,1]$

### Logistic Loss Function (Log Loss)
- **Loss for the $i^{th}$ instance:**

$$
l(y_i,\hat{y}_i)=-(y_i\log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i))
$$

### Interpretation
- The loss is small when the predicted probability matches the true class well.
- The loss becomes very large when the model is highly confident but wrong.
- This encourages the model to assign high probability to the correct class and low probability to the incorrect class.

### Decision Rule
- A predicted probability can be converted to a class label using a threshold, often 0.5.
- Changing the threshold changes the tradeoff between precision and recall.

### General Rules
- Logistic Regression models the probability of the positive class.
- The output must be interpreted together with a threshold if a hard class label is needed.
- Log loss is more appropriate than square loss for binary probability estimation.
- Logistic Regression is widely used in text classification because it works well with sparse, high-dimensional features.

## Using Logistic Regression for Bounded Regression

### Idea
- Logistic Regression naturally produces outputs in $[0,1]$.
- For bounded regression tasks, a target variable can sometimes be scaled into this interval before fitting.

### Min-Max Normalization
- A common transformation is:

$$
\frac{x - min}{max - min} \rightarrow [0,1]
$$

### Procedure
- Normalize the target into $[0,1]$.
- Train the model to predict within that range.
- Transform predictions back to the original scale after inference.

### Interpretation
- This can be useful when the target is truly bounded and predictions should remain in that range.
- It is not a general replacement for standard regression methods.

### General Rules
- Logistic-style bounded prediction is only appropriate when the target variable has a meaningful lower and upper bound.
- If the original task is standard unbounded regression, OLS or other regression models are usually more natural.
- Any target transformation used in training must be reversed consistently during evaluation and deployment.

## Logistic Regression for Sentiment Analysis

### Workflow
1. **Dataset Split:** Split the dataset $(X, y)$ into training and testing portions.
2. **Feature Construction:** Convert text documents into bag-of-words or related vectors.
3. **Model Training:** Fit Logistic Regression on the training data.
4. **Evaluation:** Measure performance on held-out data.
5. **Efficiency:** Use sparse matrices to store and process the high-dimensional text features efficiently.

### Why It Works Well
- Sentiment classification is often linearly separable enough in bag-of-words space for Logistic Regression to perform well.
- Text data usually has many dimensions but most are zero for any given document, which suits sparse linear methods.

### General Rules
- Feature extraction must be learned on training data only to avoid leakage.
- Sparse representations are essential for scaling text classification to large vocabularies.
- Logistic Regression is a strong baseline for sentiment analysis because it is simple, fast, and effective on sparse text features.

## General Rules

### Linear Models
- Linear models combine input features additively through learned weights.
- They are easy to interpret because each feature has a direct contribution to the final score.
- They often perform well when informative signals are spread across many sparse dimensions.

### Regression vs. Classification
- Use regression models when the target is continuous.
- Use classification models when the target is categorical.
- The same linear score can be used in different ways depending on the output transformation and loss function.

### Loss Functions
- The loss function defines what the model is trying to optimize.
- OLS uses square loss to penalize numeric prediction error.
- Logistic Regression uses log loss to penalize poor probability estimates for class labels.
- The choice of loss function should match the type of target and prediction task.

### Probability and Thresholds
- Logistic Regression outputs probabilities or probability-like scores.
- A classification threshold converts these probabilities into hard labels.
- Different thresholds can change precision, recall, and F1 without changing the underlying model.

### Sparse Text Features
- Text feature spaces are usually high-dimensional and sparse.
- Linear models are computationally attractive in this setting because they scale well with sparse input data.
- Sparse storage is essential for efficient memory use and training speed.

### Practical Modeling
- Always separate training and test data before fitting the model.
- Any preprocessing or feature learning that depends on the corpus should be fit on training data only.
- Evaluation should be done on held-out data to estimate generalization rather than memorization.

### Limitations
- Linear models assume additive feature effects and cannot naturally model complex feature interactions unless such interactions are encoded explicitly.
- They may miss subtle linguistic phenomena such as long-distance dependencies, compositional meaning, and nuanced word order effects.

## Glossary

### **Bias / Intercept**
- A constant term added to the weighted sum of features that shifts predictions even when all feature values are zero.

### **Linear Classifier**
- A model that predicts class-related scores from a weighted linear combination of input features, often followed by a threshold or probability transformation.

### **Linear Prediction**
- The weighted sum of input features and model coefficients, optionally plus a bias term.

### **Logistic Loss Function**
- Also known as log loss or cross-entropy loss, it penalizes wrong probability predictions and especially punishes predictions that are confident but incorrect.

### **Logistic Regression**
- A linear classification model that applies the sigmoid function to a linear score in order to predict probabilities for binary outcomes.

### **Min-Max Normalization**
- A rescaling technique used to transform values into a bounded interval, often $[0,1]$.

### **Ordinary Least Squares (OLS)**
- A linear regression method that learns parameters by minimizing the sum of squared prediction errors.

### **Sigmoid Function**
- An S-shaped function that maps any real-valued input into the interval $(0,1)$.

### **Sparse Matrix**
- A matrix representation that stores mainly non-zero entries, which is crucial for efficient text modeling.

### **Square Loss Function**
- A loss function based on the squared difference between actual and predicted numeric values.

### **Threshold**
- A cutoff used to convert predicted probabilities into discrete class labels.

---

# Code Material: Zipf's Law TF-IDF Logistic Regression

* **Summary:**
  This code material walks through an end-to-end text classification pipeline using the IMDB movie review dataset. It covers preprocessing, Zipf’s Law, vocabulary construction, TF-IDF feature extraction, sparse matrix representations, Logistic Regression training, cross-validation, and evaluation metrics, while also highlighting the general rules behind each step.

## Overview

### 1. The Text Preprocessing Pipeline
- Raw text must be converted into standardized tokens before it can be used by machine learning models.
- Typical preprocessing steps include:
  - **Tokenization:** Splitting text into smaller units such as words, punctuation marks, or other tokens.
  - **Lowercasing:** Standardizing case so equivalent forms like `Movie` and `movie` are treated the same.
  - **Stopword Removal:** Filtering out very common words that often contribute little discriminative information.
  - **Stemming or Lemmatization:** Reducing inflected or variant forms toward a more normalized base form.

#### General Rules
- Preprocessing choices directly affect the vocabulary and therefore the final feature representation.
- Different preprocessing pipelines can produce different model performance even when the classifier is unchanged.
- There is no universally best preprocessing recipe; the right choice depends on the task, corpus, and desired level of normalization.
- Stopword removal is task-dependent: words that are unhelpful for topic classification may still carry signal in sentiment or style tasks.
- Stemming is usually more aggressive and rule-based, while lemmatization is usually more linguistically grounded.

### 2. Zipf's Law and the Long Tail
- Natural language follows a highly skewed frequency distribution.
- A small number of words occur very frequently, while a very large number of words occur rarely.
- This is commonly summarized by **Zipf’s Law**, where word frequency is approximately inversely related to rank.

#### Coding Interpretation
- Sorting words by frequency reveals a few dominant terms and a large long tail of rare terms.
- A log-log plot of frequency versus rank often appears approximately linear, reflecting power-law-like behavior.

#### General Rules
- Extremely frequent words are often less discriminative because they appear in many documents.
- Extremely rare words may add noise, sparsity, misspellings, or corpus-specific artifacts.
- Vocabulary pruning is often necessary to balance information retention against computational cost.
- Replacing rare terms with a special token such as `<UNK>` is one practical way to control vocabulary size.

### 3. Feature Extraction: Bag-of-Words and TF-IDF
- Machine learning models require numeric input, so text documents must be mapped to vectors.
- A document-term representation assigns one feature dimension per vocabulary term.
- Common weighting schemes include:
  - **Binary:** Whether a term appears.
  - **Term Frequency (TF):** How many times a term appears in the document.
  - **TF-IDF:** A weighted version that upweights terms frequent in one document but infrequent in the corpus.

#### General Rules
- Feature extraction is part of the model design, not just a formatting step.
- The same text can produce very different vectors depending on tokenization, normalization, vocabulary filtering, and weighting.
- TF emphasizes repeated terms inside a document.
- IDF reduces the influence of terms that appear in many documents.
- TF-IDF is often a strong baseline for text classification because it balances document-specific frequency with corpus-wide rarity.

### 4. Sparse Matrices
- Text feature spaces are typically high-dimensional because the vocabulary can contain thousands or tens of thousands of terms.
- However, each individual document usually contains only a small subset of the vocabulary.
- As a result, document-term matrices are mostly zeros.

#### General Rules
- Sparse matrix formats store only non-zero entries and their locations.
- Sparse storage is essential for efficient memory usage and fast computation in bag-of-words and TF-IDF pipelines.
- Many linear models used in NLP are designed to work efficiently with sparse feature matrices.

### 5. Logistic Regression for Text Classification
- Logistic Regression is a linear classifier that maps a weighted feature vector to a probability through the sigmoid function.
- It is commonly used for binary sentiment analysis because it works well with sparse, high-dimensional features.

#### General Rules
- Logistic Regression learns a weight for each feature and an intercept term.
- Positive weights push predictions toward the positive class; negative weights push predictions toward the negative class.
- Logistic Regression predicts probabilities, which can be converted to labels using a decision threshold.
- It is often a strong baseline for text classification because it is simple, interpretable, and effective.

### 6. Model Evaluation
- A classifier should be evaluated on held-out data rather than only on the training set.
- Common evaluation metrics include:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **Confusion Matrix**

#### General Rules
- Accuracy is intuitive but can be misleading when classes are imbalanced.
- Precision matters more when false positives are costly.
- Recall matters more when false negatives are costly.
- F1 is useful when precision and recall must both be considered.
- The confusion matrix provides the raw counts behind these metrics.

## Python Libraries

### pandas
- Used for loading and manipulating tabular datasets such as CSV files.
- A `DataFrame` stores rows and columns and is convenient for handling labeled data.
- Common uses here include reading the IMDB data and accessing review text and sentiment labels.

### nltk
- Used for tokenization, stopword lists, and stemming.
- Provides reusable components for building text preprocessing pipelines.

### collections.defaultdict
- Useful for counting frequencies without manually checking whether a key already exists.

### re
- Used for regular expression-based text cleaning, such as removing non-alphanumeric characters.

### matplotlib
- Used for visualizing distributions, such as Zipf plots and histograms of document lengths.

### numpy
- Used for numerical summaries and array operations.

### tqdm
- Used to monitor progress in long-running loops.

### math
- Used for functions such as logarithms in TF-IDF computation.

### sklearn
- Used for model training, feature extraction, splitting data, cross-validation, and evaluation.

### scipy
- Used for sparse matrix data structures such as CSR matrices.

### pickle
- Commonly used to save trained models or preprocessing objects for later reuse.

## Sentiment Analysis on the IMDB Dataset

### Dataset Loading
```python
import pandas as pd

df = pd.read_csv('movie_data.csv')

print(len(df))
```

```text
50000
```

### Interpretation
- The dataset contains 50,000 movie reviews with associated sentiment labels.
- Each row consists of raw review text and a binary sentiment target.

#### General Rules
- Text classification datasets typically contain raw text plus labels.
- Before training a model, the text must be transformed into features while labels remain in their original supervised form.

## NLTK-Based Preprocessing

### Stopwords and Tokenization
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
```

### Case Sensitivity Example
```python
print('An' in stop)
print('an' in stop)
print("'s" in stop)
```

#### General Rules
- Python string membership is case-sensitive.
- If stopword matching is done after lowercasing, `An` and `an` can be treated consistently.
- Whether punctuation-like fragments such as `"'s"` are removed depends on the tokenizer and stopword list, not just on the word itself.

### Preprocessing Function
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

#### General Rules
- A preprocessing pipeline usually combines multiple operations in a fixed order.
- The order matters: tokenization must happen before token-level filtering or stemming.
- Lowercasing before stopword filtering improves consistency.
- Stemming reduces feature sparsity by merging related surface forms, but may reduce interpretability.
- A preprocessing function should be applied consistently to both training and inference text.

## Testing Zipf's Law

### Counting Frequencies
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
- The code constructs a corpus-wide frequency dictionary.
- Regex-based cleaning removes non-alphanumeric punctuation before splitting on whitespace.

#### General Rules
- Corpus statistics depend strongly on preprocessing choices.
- Different tokenization or cleaning rules lead to different frequency counts and different Zipf plots.
- Frequency counts are corpus-level summaries, not document-level features.

### Frequency Ranking
```python
order_tokens = sorted(list(freq.items()), key=lambda x: -x)[1]
```

#### General Rules
- Sorting by descending frequency reveals the head and tail of the distribution.
- The most frequent words are usually function words or highly common content words.
- The long tail often consists of rare names, misspellings, domain-specific words, and one-off tokens.

### Visualization
```python
import matplotlib.pyplot as plt

y = [freq for token, freq in order_tokens]
plt.loglog(y)
```

#### General Rules
- A log-log visualization is useful when frequency values span multiple orders of magnitude.
- Raw linear-scale plots can obscure the tail because high-frequency words dominate the vertical scale.
- Histograms of review lengths are useful for understanding variability in document size, which can influence TF weighting and normalization.

## Building TF-IDF Ourselves

### Document Frequency
```python
from tqdm import tqdm
DF = defaultdict(float)
for doc in tqdm(df.review):
    tokens = pre_processing_by_nltk(doc)
    for token in set(tokens):
        DF[token] += 1
```

#### General Rules
- **Document Frequency (DF)** counts how many documents contain a term, not how many total times the term appears in the corpus.
- Using `set(tokens)` is important because a document should contribute at most one count to DF for each term.
- DF is corpus-dependent and must be recomputed if the corpus changes.

### Vocabulary and IDF Construction
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

#### General Rules
- Frequency thresholds are a practical way to prune rare terms from the vocabulary.
- Vocabulary pruning trades off coverage against sparsity, noise, and computational cost.
- IDF is higher for terms appearing in fewer documents and lower for terms appearing in many documents.
- The exact IDF formula may vary across implementations, but the core idea is always to downweight common terms.

### Handling Unknown Words
```python
IDF['<UNK>'] = 1
vocab['<UNK>'] = len(vocab)
```

#### General Rules
- Unknown-token handling provides a fallback for terms not seen in the retained vocabulary.
- Mapping rare or unseen terms to `<UNK>` can improve robustness and control vocabulary growth.
- The choice of `<UNK>` weighting is a modeling decision and may influence downstream results.

### TF-IDF Feature Function
```python
def tfidf_feature_extractor(doc, vocab, IDF):
    tokens = pre_processing_by_nltk(doc)
    for i, token in enumerate(tokens):
        if token not in vocab:
            tokens[i] = '<UNK>'
    TF = defaultdict(int)
    for token in tokens:
        TF[token] += 1
    x =  * len(vocab)
    for token in set(tokens):
        tfidf = log(TF[token] + 1) * IDF[token]
        token_id = vocab[token]
        x[token_id] = tfidf
    return x
```

#### General Rules
- TF-IDF combines document-level term frequency with corpus-level document rarity.
- Log-scaled TF reduces the influence of repeated occurrences.
- A dense Python list is conceptually simple but inefficient for high-dimensional sparse text features.
- Manual TF-IDF implementations are useful for learning the logic, but production pipelines typically use library implementations for efficiency and consistency.

## Train/Test Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size=0.2, shuffle=True
)
```

#### General Rules
- Train/test splitting is used to estimate generalization to unseen data.
- The test set should remain unseen during model training and model selection.
- Shuffling helps reduce ordering bias when the dataset has no meaningful sequence structure.
- Any vocabulary learning or IDF estimation should ideally be performed on training data only to avoid leakage.

## Logistic Regression Training

### Small Training Subsets
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train[:1000], y_train[:1000])
clf.score(X_test, y_test)
```

#### General Rules
- Increasing the amount of training data often improves model performance, though gains may diminish.
- Small subsets are useful for debugging, but full training usually produces stronger generalization if computation allows.

### Convergence Warnings
```python
clf = LogisticRegression(random_state=0).fit(X_train[:10000], y_train[:10000])
```

#### General Rules
- A convergence warning means the optimizer hit its iteration limit before fully converging.
- This does not always make the model unusable, but it suggests that optimization settings may need adjustment.
- Common responses include increasing `max_iter`, changing the solver, or improving feature scaling when appropriate.

## Sparse Matrix Training

### CSR Conversion
```python
import scipy.sparse as sparse
sparse_X = sparse.csr_matrix(X_train)
```

### Why It Helps
- CSR format stores only non-zero feature values and their positions.
- This is especially efficient for document-term matrices in NLP.

#### General Rules
- Sparse matrix formats are usually preferred over dense matrices for large text datasets.
- Efficiency gains can be substantial in both memory usage and model fitting time.
- Many scikit-learn text-processing tools already return CSR matrices by default.

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

#### General Rules
- `CountVectorizer` converts raw documents into a sparse document-term matrix of token counts.
- If no vocabulary is supplied, the vocabulary is learned during `fit`.
- The learned vocabulary maps terms to integer feature indices.
- The exact tokens included depend on preprocessing, tokenization rules, case handling, and frequency settings.

## TF-IDF with Library Tools

### TfidfTransformer
```python
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
```

#### General Rules
- `TfidfTransformer` takes count features and reweights them into TF-IDF features.
- `smooth_idf=True` prevents zero-division-style edge cases and slightly adjusts IDF estimation.
- `norm='l2'` normalizes each document vector to unit L2 length, which reduces the effect of document length on vector magnitude.

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

#### General Rules
- `TfidfVectorizer` combines tokenization, vocabulary learning, counting, and TF-IDF weighting in one object.
- It is conceptually equivalent to `CountVectorizer` followed by `TfidfTransformer`.
- Fitting learns vocabulary and IDF statistics from the training corpus.
- Transforming new text applies the learned mapping without rebuilding the vocabulary.
- Custom tokenizers allow integration of task-specific preprocessing pipelines.

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

#### General Rules
- `LogisticRegressionCV` performs internal cross-validation to select regularization settings automatically.
- Cross-validation helps estimate model quality more robustly than a single split on the training data.
- The scoring metric used in model selection affects which model configuration is chosen.
- Internal cross-validation should still be separated from the final test evaluation.

## Predictions and Probabilities

```python
yhat = clf.predict(X_test)
clf.predict_proba(x_ins)
```

#### General Rules
- `predict` returns discrete class labels.
- `predict_proba` returns estimated class probabilities.
- Predicted probabilities support threshold-based decision making and confidence-aware analysis.
- The probability outputs correspond to the learned classes in class order.

## Confusion Matrix and Metrics

```python
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, yhat)
```

### Manual Metric Calculation
```python
prec = confusion_mat / (confusion_mat + confusion_mat)[1]
rec = confusion_mat / (confusion_mat + confusion_mat)[1]
f1 = 2 * prec * rec / (prec + rec)
```

#### General Rules
- Precision measures the fraction of predicted positives that are correct.
- Recall measures the fraction of actual positives that are recovered.
- F1 combines precision and recall through their harmonic mean.
- Manual calculations are useful for understanding metric definitions.
- Library functions are preferred in practice because they are less error-prone and easier to extend.

## Example Inference

```python
text = 'this is my favorite movie!'
x_ins = tfidf.transform([text])
clf.predict(x_ins)
clf.predict_proba(x_ins)
```

#### General Rules
- Inference on new text must reuse the same preprocessing and feature-extraction pipeline learned during training.
- New text is transformed into the same feature space as the training data.
- Words unseen during fitting are ignored or handled according to the vectorizer’s vocabulary rules.
- A trained text classifier can output both a label and a probability for new user-provided text.

## Global Rules from This Notebook

### Pipeline Rules
- A complete text classification workflow usually includes:
  1. Load labeled text data
  2. Preprocess text
  3. Build vocabulary and features
  4. Split data
  5. Train model
  6. Evaluate on held-out data
  7. Apply the trained pipeline to new text

### Representation Rules
- Vocabulary design, frequency filtering, and weighting strongly affect model behavior.
- TF-IDF usually improves over raw counts when common terms would otherwise dominate.
- Sparse representations are standard for high-dimensional text data.

### Modeling Rules
- Logistic Regression is a strong baseline for binary text classification.
- More training data often helps, but optimization settings and representation choices also matter.
- Convergence warnings indicate optimization issues, not necessarily total model failure.

### Evaluation Rules
- Test data should remain separate from training and tuning steps.
- Confusion-matrix-based metrics provide more detailed information than accuracy alone.
- Probability outputs are useful for calibration, thresholding, and downstream decision-making.

## Glossary

### **CSR Matrix**
- A compressed sparse row matrix format that stores only non-zero entries and is efficient for large sparse text data.

### **Document Frequency (DF)**
- The number of documents in a corpus that contain a term at least once.

### **IDF**
- A weighting factor that downweights terms appearing in many documents and upweights terms appearing in fewer documents.

### **Logistic Regression**
- A linear classification model that maps weighted feature sums into probabilities through a sigmoid-like formulation.

### **Long Tail**
- The large set of low-frequency terms that appear rarely in a corpus.

### **TF**
- A term-frequency measure reflecting how often a term appears in a document.

### **TF-IDF**
- A combined weighting scheme that highlights terms that are frequent in a document but relatively uncommon in the corpus.

### **TfidfTransformer**
- A tool that converts count-based document-term matrices into TF-IDF-weighted matrices.

### **TfidfVectorizer**
- A tool that directly converts raw text documents into TF-IDF feature matrices.

### **Unknown Token (`<UNK>`)**
- A fallback symbol used to represent rare or unseen terms that are not kept as separate vocabulary entries.

### **Zipf's Law**
- An empirical distributional pattern in language where a word’s frequency decreases roughly as its rank increases.

---
