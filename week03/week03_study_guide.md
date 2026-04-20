Here is your comprehensive Week 03 cheat sheet, organized by priority based on the historical quiz patterns from Weeks 01 and 02.

***

# Week 03 — Quiz Cheat Sheet
## Word Representations, Embeddings & Word2Vec

***

## 🔴 Tier 1 — Must Know (Highest Quiz Probability)

### Sparse vs. Dense Representations

| Property | Sparse (TF-IDF / BoW) | Dense (Word2Vec / GloVe) |
|---|---|---|
| Dimensions | \(\|V\|\) or \(\|D\|\) — 10K–50K+ | 50–1000 |
| Entry values | Counts / weights (≥ 0) | Real-valued, positive or negative |
| Interpretability | Each dim = a specific word | Dims have **no** direct label |
| Generalization | Poor — no sharing between words | Better — similar words → similar vectors |
| Most entries | Zero | Non-zero |

**Quiz trap:** TF-IDF is sparse and vocabulary-indexed. Word2Vec is dense. Neither is "always better." [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Distributional Hypothesis ⚠️ *Named in learning objectives*

> **"You shall know a word by the company it keeps."** — J.R. Firth, 1957

- Words appearing in **similar contexts** tend to have **similar meanings** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- **Similarity ≠ Synonymy:** `good`/`bad` score 0.72 (antonyms, identical contexts), `dog`/`cat` are similar but not synonyms [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- **Limitation:** distributional models cannot inherently distinguish antonyms from synonyms [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### CBOW vs. Skip-Gram ⚠️ *Learning objective explicitly lists this*

| Feature | CBOW | Skip-Gram |
|---|---|---|
| **Input** | Context words \(c_1 \ldots c_m\) | One center word \(w\) |
| **Predicts** | Missing center word | Surrounding context words |
| **Probability** | \(P(w \mid c_1, \ldots, c_m)\) | \(P(c \mid w)\) |
| **Intuition** | "Fill in the blank from context" | "Predict neighbors from one word" |
| **Strength** | Faster; better for frequent words | Better for rare words; more relationships |

**Quiz trap:** They are **inverse tasks.** CBOW = many-in, one-out. Skip-Gram = one-in, many-out. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Skip-Gram Objective

Minimize average negative log-likelihood:
\[J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \leq j \leq m \\ j \neq 0}} \log P(w_{t+j} \mid w_t;\, \theta)\]

Context probability via softmax:
\[P(o \mid c) = \frac{\exp(\mathbf{u}_o^T \mathbf{v}_c)}{\sum_{w \in V} \exp(\mathbf{u}_w^T \mathbf{v}_c)}\]

- \(\mathbf{v}_c\) = center-word vector; \(\mathbf{u}_o\) = context-word vector [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- Each word has **two** vectors: one as center (\(\mathbf{v}\)), one as context (\(\mathbf{u}\)) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- Total parameters = \(2d|V|\) — e.g., \(d=300\), \(|V|=10{,}000\) → **6,000,000 parameters**  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Negative Sampling

**Why:** Full softmax sums over entire vocabulary — too slow [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

**What it does:** Replaces softmax with binary classification — distinguish real context words (positive) from random noise words (negative) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

**Loss function:**
\[L_{CE} = -\left(\log \sigma(c_{\text{pos}} \cdot w) + \sum_{i=1}^{k} \log \sigma(-c_{\text{neg}_i} \cdot w)\right)\]

- \(k\) = number of negative samples; typical values **3–10** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- \(\sigma(\cdot)\) = sigmoid function mapping dot product to probability [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

**Training intuition:** move `apricot` closer to `jam`; push it away from `aardvark` and `Tolstoy` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

**Quiz trap:** The minus sign in \(\sigma(-c_{\text{neg}} \cdot w)\) is **inside** the sigmoid argument, not outside the log — makes high dot product → low positive probability → pushes words apart [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Cosine Similarity

\[\text{cosine\_similarity}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}\]

- Range: **-1** (opposite) to **+1** (identical direction) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- `wv.similarity()` computes this automatically [codecademy](https://www.codecademy.com/learn/dscp-natural-language-processing/modules/dscp-word-embeddings/cheatsheet)

***

### Key Similarity Values to Know

| Pair | Score | Key Lesson |
|---|---|---|
| `good` / `bad` | **0.72** | Antonyms score very high — same contexts |
| `king` / `queen` | 0.65 | Semantically related |
| `king` / `King` | 0.52 | Same concept, different case — model is case-sensitive |
| `hate` / `like` | 0.39 | Verb antonyms score lower than adjective antonyms |
| `car` / `cereal` | ~0.13 | Completely unrelated → near zero |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## 🟡 Tier 2 — High Priority (Likely Tested)

### PMI — Pointwise Mutual Information

\[\text{PMI}(w, c) = \log_2 \frac{P(w, c)}{P(w) \cdot P(c)}\]

| PMI Value | Meaning |
|---|---|
| \(> 0\) | Co-occurs more than expected by chance → informative |
| \(= 0\) | Exactly as expected by chance → no association |
| \(< 0\) | Less than expected → typically set to 0 (PPMI) |

**Why PMI beats raw frequency:** `drink`+`it` has frequency 3 but PMI 1.25 (generic). `drink`+`tea` has frequency 2 but high PMI (specific association). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

**PPMI (Positive PMI):** Replace negative PMI values with 0 — used because negative values are unreliable with sparse data [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Term-Document vs. Term-Term Matrix

| Property | Term-Document | Term-Term |
|---|---|---|
| Rows | Vocabulary words | Vocabulary words |
| Columns | Documents | Context words |
| Context scope | Entire document | Local sliding window |
| Related method | TF-IDF | PMI-weighted co-occurrence |
| Problem | Sparse, biased toward function words | Same sparsity problem |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Analogy Arithmetic

\[\text{vector}(\text{king}) - \text{vector}(\text{man}) + \text{vector}(\text{woman}) \approx \text{vector}(\text{queen})\]

- Result is **approximate**, not exact [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- Confirmed similarity to `woman` vector: **0.719** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- Gender offset: cosine similarity of `(queen−king)` and `(woman−man)` = **0.449** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- Unrelated offset: cosine similarity of `(queen−king)` and `(moon−sun)` ≈ **−0.068** (no relationship) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Pre-trained Embeddings — Fast Facts

| Model | Dims | Vocab | Training Data | Method |
|---|---|---|---|---|
| Google News Word2Vec | 300 | 3M words & phrases | ~100B words, news | Skip-Gram + neg. sampling |
| GloVe 6B | 50/100/200/300 | ~400K uncased | Wikipedia + Gigaword, ~6B tokens | Co-occurrence matrix factorization |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

**Quiz trap:** Word2Vec uses a **predictive neural** objective. GloVe uses **global co-occurrence matrix factorization**. Both produce dense embeddings — different methods, similar quality. [datascientistinsights.substack](https://datascientistinsights.substack.com/p/glove-vs-word2vec-in-practice-does)

***

### doc2vec by Averaging — Critical Failure Case

```python
cos_sim("I like this course so much", "I do not like this course") = 0.931
cos_sim("I like this movie",          "I hate this movie")         = 0.901
```

**Why this fails:** Averaging washes out negation (`not`) and conflates antonyms (`like`/`hate` — similarity 0.39 but shared context words dominate the mean) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

**Lesson:** Averaged embeddings ≠ good sentiment representations. Same fundamental weakness as bag-of-words from Week 02. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/a311cb5c-3f50-4522-9108-876314359d2d/week02_content.md)

***

## 🟢 Tier 3 — Know the Concept (Supporting Detail)

### Key gensim Functions

| Function | What It Does | Returns |
|---|---|---|
| `wv['word']` | Look up a word vector | `numpy.ndarray` (shape: `(300,)`) |
| `wv.similarity(a, b)` | Cosine similarity between two words | `float` |
| `wv.most_similar(positive=['word'], topn=N)` | Top-N nearest neighbors | List of `(word, score)` tuples |
| `wv.doesnt_match([list])` | Word most unlike group centroid | `str` |
| Missing word → | Raises `KeyError` — wrap in `try/except` | — |

 [radimrehurek](https://radimrehurek.com/gensim/models/word2vec.html)

***

### GloVe Loading Workflow

```python
# Convert GloVe format → Word2Vec format, then load
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
```
After loading, `model` supports same `.similarity()`, `.most_similar()` interface as Word2Vec [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Gradient Intuition (Conceptual, Not Computation)

- \(\sigma(c_{\text{pos}} \cdot w) - 1 < 0\) always → positive context pushed **toward** target [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- \(\sigma(c_{\text{neg}} \cdot w) > 0\) always → negative context pushed **away** from target [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- Learning rate \(\eta\) controls step size in SGD updates [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## ⚡ Absolute Wording Traps *(Based on Week 02 Quiz Q2 pattern)*

Based on the instructor's established pattern of using absolute words as false-answer traps: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/fdafe58b-519a-495a-bb9c-c7b79a0c4818/quiz02.md)

| Statement | Verdict |
|---|---|
| "More complex TF-IDF/embedding variants are **always** better" | ❌ False |
| "Word2Vec embeddings **must** outperform TF-IDF" | ❌ False |
| "Distributional similarity **always** means synonymy" | ❌ False |
| "Pre-trained embeddings are **always** better than custom ones" | ❌ False |
| "Negative sampling changes the model **architecture**" | ❌ False — changes training objective only |
| "Training Word2Vec **requires** labeled data" | ❌ False — self-supervised |
| "PMI is **always** positive" | ❌ False — can be negative; PPMI clips to 0 |

***

## 📝 Predicted Quiz Questions

Based on escalating computation pattern (Week 01: code output, Week 02: formula calculation, Week 03: likely both):

1. **Code output:** *What does `wv['king']` return? What type?* → `numpy.ndarray`, length 300 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
2. **Definition:** *What is the distributional hypothesis?* → Words in similar contexts have similar meanings (Firth, 1957) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
3. **Contrast:** *CBOW predicts ___ from ___; Skip-Gram predicts ___ from ___.* [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
4. **Formula computation:** *Given \(|V|=5{,}000\) and \(d=100\), how many parameters does Skip-Gram learn?* → \(2 \times 100 \times 5{,}000 = 1{,}000{,}000\)  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
5. **Formula computation:** *Compute \(\text{PMI}(w,c)\) given \(P(w,c)=0.01\), \(P(w)=0.1\), \(P(c)=0.2\).* → \(\log_2(0.01 / 0.02) = \log_2(0.5) = -1\) → PPMI = 0 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
6. **True/False select-all:** *Which statements about TF-IDF and Word2Vec are true?* — watch for "always" and "must" traps [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/fdafe58b-519a-495a-bb9c-c7b79a0c4818/quiz02.md)
7. **Conceptual explanation:** *Why does averaging word vectors fail for sentiment classification?* → negation washed out; antonyms have similar vectors [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
8. **Tool name:** *Which Python library is used to load and query Word2Vec models?* → `gensim` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
9. **Why:** *Why is negative sampling used?* → avoid full softmax over vocabulary on each update [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
10. **Interpretation:** *`good` and `bad` have similarity 0.72. Does this mean they are synonyms?* → No — distributional similarity ≠ synonymy; antonyms share contexts [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

