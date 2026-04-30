### CRITICAL ANALYSIS AND NOTE IMPROVEMENT: WEEK 04

Based on the testing patterns identified in Week 03 (specifically the focus on 
hyperparameter quality, code logic, and precise mathematical definitions), your 
Week 04 study guide is missing several "Tier 1" technical details. 

The instructor previously tested the "factors affecting quality" (Week 03 Q2); 
therefore, the specific limitations of N-gram orders and the "Data vs. Method" 
tradeoff are high-risk areas.

---

### IMPROVED WEEK 04 STUDY GUIDE (ENHANCED SECTION)

#### 1. Language Model Fundamentals & Applications
* **Formal Definition:** A probability distribution over sequences of words 
    $P(w_1 \ldots w_n)$.
* **Primary Functions:**
    1.  **Evaluation:** Scoring existing text for fluency.
    2.  **Generation:** Producing new text by sampling from the distribution.
* **Key Applications:** Used in Speech Recognition (Acoustic + Language 
    Model), Machine Translation, and Summarization.

#### 2. The N-Gram Architecture & Markov Assumption
* **Chain Rule:** Used to generate words left-to-right.
* **Markov Assumption:** Simplifies $P(w | \text{history})$ by only 
    conditioning on the immediate $n-1$ preceding words.
* **Probability Estimation (MLE):** Calculated via counts: 
    $P(\text{door} | \text{the}) = \frac{\text{Count}(\text{the door})}{\text{Count}(\text{the *})}$.
* **N-gram Order Effects:**
    * **Higher orders ($n > 3$):** Capture local dependencies like idioms, 
        morphology, and world knowledge.
    * **Critical Weakness:** Cannot capture long-distance dependencies.

#### 3. Evaluation: Likelihood and Perplexity
* **Shannon's Game:** Evaluating a model by its ability to predict the next 
    word.
* **Log Likelihood:** $\sum_{w \in X} \log P(w|\theta)$.
* **Perplexity ($PP$):** The "average per-word branching factor".
    * **Formula:** $\text{exp}(-\frac{1}{|X|} \sum \log P(x_i))$.
    * **Goal:** Lower perplexity indicates a "less surprised" (better) model.

#### 4. Sparsity and Smoothing (The "Quality" Tier)
* **Sparsity Problem:** As $n$ increases, the fraction of $n$-grams seen in 
    the corpus drops sharply, leading to zero-count errors.
* **Linear Interpolation:** Combines different orders to balance "specific 
    but sparse" (Higher-order) with "dense but general" (Lower-order).
    * **Weights ($\lambda$):** Chosen via grid search or EM on **held-out 
        data**.
* **Absolute Discounting:** Subtracts a constant $d$ from counts and 
    redistributes mass to new events.
* **Kneser-Ney (KN) Smoothing (Tier 1 Priority):**
    * Uses **Context Fertility:** The number of distinct context types a word 
        appears in.
    * Logic: A high-frequency word with low fertility (e.g., "Francisco") is 
        less likely in new contexts than a fertile word.

#### 5. Strategic Optimization: Data vs. Method
* **The "Big Data" Rule:** Having more training data generally improves 
    performance more than a better algorithm (e.g., 10M Katz > 100k KN).
* **Algorithm Superiority:** For a fixed amount of data, Kneser-Ney 
    consistently outperforms Katz/Smoothing-linear estimators.

---

SECTION 1: CONCEPTUAL FOUNDATIONS

1. What is the formal definition of a language model?

(A) A list of all grammatical rules for a specific language.

**(B) A probability distribution over sequences of words or sentences.**

(C) A database of word embeddings and their semantic meanings.

(D) A software tool used exclusively for spell-checking.

Rationale: LMs assign probabilities $P(w_1, \ldots, w_n)$ to sequences to determine how likely they are to occur.

2. Which of the following are primary applications of language models?

**(A) Speech recognition, Machine translation, and Summarization.**

(B) Sorting algorithms and database indexing.

(C) Hardware optimization and GPU thermal management.

(D) Only generative tasks like writing poetry.

3. What is the Markov assumption in $n$-gram language models?

(A) Every word in a sentence is independent of every other word.

(B) The probability of a word depends on all preceding words in the document.

**(C) The next word is predicted using only a limited number of previous words.**

(D) Language follows a hidden Markov chain that never changes over time.

Rationale: Trigrams assume $P(w_i | w_{i-1}, w_{i-2})$, ignoring the history further back to simplify calculation.

4. How are $n$-gram probabilities typically estimated?

(A) From hand-written grammar rules.

**(B) From counts of $n$-grams in a large corpus.**

(C) Using only part-of-speech tags.

(D) By surveying native speakers for their intuition.

5. Why do higher-order $n$-gram models (e.g., 4-grams) often perform better than lower-order models (e.g., bigrams)?

(A) They require significantly less memory.

(B) They are less likely to suffer from sparsity.

**(C) They capture more contextual dependencies and local knowledge.**

(D) They eliminate the need for any smoothing.

SECTION 2: EVALUATION AND SPARSITY

6. What is the "sparsity" problem in language modeling?

(A) Having too much data for the computer to process.

**(B) Many valid $n$-grams are unseen or rare in training, leading to zero-count errors.**

(C) The vocabulary is too small to describe complex concepts.

(D) The model predicts words that are too simple.

7. What is the primary purpose of smoothing?

(A) To remove rare words from the training corpus.

**(B) To redistribute probability mass so unseen $n$-grams have non-zero probability.**

(C) To increase all observed counts equally.

(D) To decrease the size of the model.

Rationale: Smoothing "flattens" the distribution so the model generalizes better to new text.

8. Which statement about Perplexity ($PP$) is correct?

(A) Higher perplexity always means a better, more "certain" model.

(B) Perplexity is calculated only on the training set.

**(C) Lower perplexity indicates a "less surprised" (better) model on test data.**

(D) Perplexity is the same as the total number of words in the vocabulary.

9. Why should hyperparameters (like smoothing weights) be tuned on held-out data?

**(A) To ensure the test set remains an unbiased final evaluation.**

(B) Because held-out data is always larger than the test set.

(C) Because likelihood cannot be computed on the test set.

(D) Because smoothing only works on validation sets.

SECTION 3: ADVANCED TECHNIQUES (TIER 1)

10. What makes Kneser-Ney (KN) smoothing different from simpler methods?

(A) It uses neural network layers.

**(B) It uses context fertility (number of distinct contexts a word appears in).**

(C) It relies strictly on raw frequency counts.

(D) It only works for unigram models.

11. Which describes the difference between a unigram and a trigram?

(A) A unigram uses two previous words; a trigram uses none.

**(B) A unigram uses no previous context; a trigram uses two previous words.**

(C) A unigram is always more accurate than a trigram.

(D) A unigram requires smoothing, while a trigram does not.

12. Why is conditioning on the full left context difficult?

(A) Because the vocabulary is always closed.

(B) Because the chain rule is invalid for long sentences.

**(C) Because the number of possible histories is too large to estimate reliably.**

(D) Because language models only work right-to-left.

13. What is the main weakness of raw Maximum Likelihood Estimates (MLE)?

(A) They require expensive hardware.

**(B) They assign zero probability to $n$-grams not seen in training.**

(C) They ignore the counts in the training data.

(D) They cannot be computed via simple division.

14. What is the trade-off for using higher-order $n$-grams?

(A) They capture less context but are more dense.

**(B) They capture more context but suffer significantly more from sparsity.**

(C) They make the model smaller but more complex.

(D) They eliminate the need for a training corpus.

15. Which of the following best describes Linear Interpolation?

(A) Choosing the single highest-order $n$-gram and ignoring others.

**(B) Mixing probabilities from multiple $n$-gram orders (e.g., trigram + bigram + unigram).**

(C) Using word embeddings instead of counts.

(D) Subtracting a fixed constant from every count.

16. What is the core mechanism of Absolute Discounting?

(A) Multiplying all counts by a fixed percentage.

(B) Removing all words that appear fewer than 5 times.

**(C) Reducing observed counts by a constant $d$ and redistributing that mass.**

(D) Setting all counts to 1 to ensure uniformity.

SECTION 4: PREDICTED QUIZ TRAPS

StatementVerdictCorrect Logic"Hyperparameters should be tuned on the test set."FALSEUse held-out/validation data to avoid "cheating.""A PP of 100 is better than a PP of 10."FALSELower perplexity is better."$N$-grams track long-range document context."FALSEThey only track local correlations (within the $n$ window)."KN smoothing uses raw frequency for backoff."FALSEIt uses fertility (the variety of contexts)."A better algorithm always beats more data."FALSEPer the Goodman Curve, $10\times$ more data usually beats a better method.SECTION 5: QUANTITATIVE PRACTICE

Q1. Probability Calculation

Given a trigram model, how is the sequence "please close the door" decomposed?

Answer: $P(\text{please}) \cdot P(\text{close}|\text{please}) \cdot P(\text{the}|\text{please, close}) \cdot P(\text{door}|\text{close, the})$

Q2. Perplexity Math

If a language model assigns a probability of $0.01$ to every word in a test set, what is the perplexity?

Answer: 100 (Perplexity is the inverse of the probability in a uniform distribution: $1/0.01 = 100$).

Q3. Absolute Discounting Numerator

If "please close" appears 10 times and $d=0.75$, what is the adjusted count before redistribution?

Answer: 9.25 ($10 - 0.75 = 9.25$).

