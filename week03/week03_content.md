# Week 03 
## Overview
* **Summary:**In English, we represent words as a sequence of characters. In computational models, we use mathematical representations. In this series of lectures, we will learn how we can generate mathematical representations of meanings of words.
* After this week, you will be able to:
    * Outline how to represent words in NLP systems
    * Define lemmas, senses of words, and word relations
    * Define the distributional hypothesis
    * Distinguish between TF-IDF and Word2Vec embeddings
    * Identify the semantic properties of embeddings
    * Contrast neural network architectures that are used to learn word representations:
        * Skip-gram vs. Continuous Bag of Words

## Lecture Content
* Dense Vector Representations
* Vector Semantics and Embeddings 
* Interesting Properties/Applying Word Embeddings

## Code Content
* word2vec notebook

---

Here is the full Section 1 with all improvements **inserted inline** — nothing removed, everything original preserved, with additions clearly marked as **⬆ ADDED** so you know exactly what's new.

***

## Lecture Content: Dense Vector Representations

### Overview
This lecture introduces word embeddings as dense vector representations of words. It contrasts them with traditional sparse representations and explains how CBOW, Skip-Gram, and negative sampling are used to learn useful word vectors.

### Outline
- Traditional distributional representation
- Dense vector representations
- Interesting properties of word vectors
- Applying word embeddings

***

## Traditional Distributional Representation

### Previous Word Representation
Traditional word representations use sparse, high-dimensional vectors.

- Dimensions are often \(|V|\) or \(|D|\)
- \(|V|\) is typically in the tens of thousands
- Dimensions are generally not interpretable

> ⬆ **ADDED — Quiz context:** Examples of sparse representations include **TF-IDF vectors** and **bag-of-words count vectors** from Week 02. Each dimension corresponds to a specific vocabulary word, so the vector size equals the vocabulary size. Because any given document uses only a small fraction of all words, most entries are zero. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Word Embeddings

### Dense Vector Representation
Word embeddings represent words as dense vectors instead of sparse vectors.

- Short vectors, often 50 to 1000 dimensions
- Real-valued entries, which may be positive or negative
- Often perform better than sparse representations
- Can improve generalization because they use fewer dimensions

> ⬆ **ADDED — Sparse vs. Dense Quick-Reference Table:**

| Property | Sparse (TF-IDF / BoW) | Dense (Word2Vec) |
|---|---|---|
| Dimensionality | \(\|V\|\) or \(\|D\|\), often 10K–50K+ | 50–1000 |
| Entry values | Counts or weights (≥ 0) | Real-valued, positive or negative |
| Interpretability | Each dim = a specific word | Dims are **not** directly interpretable |
| Generalization | Poor — no sharing between similar words | Better — similar words have similar vectors |
| Most entries | Zero (sparse) | Non-zero (dense) |

 [slpcourse.github](https://slpcourse.github.io/materials/lecture_notes/Lecture_12_Word_embedding.pdf)

> ⬆ **ADDED — Quiz trap:** TF-IDF and Word2Vec are **not** the same type of representation. TF-IDF is sparse and vocabulary-indexed. Word2Vec is dense and learns continuous vectors. Neither is "always better" — the best choice depends on the task and data. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Two Learning Views

### Translation 1: Predict the Missing Word
Given context words, predict the missing word.

- Continuous Bag of Words (CBOW)
- \(P(w \mid c_1, c_2, \ldots, c_m)\)

### Translation 2: Predict the Context
Given a word, predict its surrounding context words.

- Skip-Gram
- \(P(c \mid w)\)

> ⬆ **ADDED — CBOW vs. Skip-Gram Direct Contrast** *(learning objective explicitly lists this)*:

| Feature | CBOW | Skip-Gram |
|---|---|---|
| **Input** | Multiple context words | One center word |
| **Predicts** | The missing center word | Surrounding context words |
| **Probability** | \(P(w \mid c_1, \ldots, c_m)\) | \(P(c \mid w)\) |
| **Intuition** | "Fill in the blank from context" | "Given the word, predict its neighbors" |
| **Relative strength** | Faster; better for frequent words | Better for rare words; captures more linguistic relationships |

 [pinecone](https://www.pinecone.io/learn/series/nlp/dense-vector-embeddings-nlp/)

> ⬆ **ADDED — Quiz trap:** CBOW and Skip-Gram are **inverse** tasks. CBOW uses many inputs to predict one output. Skip-Gram uses one input to predict many outputs.

***

## Skip-Gram

### Core Idea
Skip-Gram learns by using a center word to predict nearby context words. This is a self-supervised learning setup.

> ⬆ **ADDED:** "Self-supervised" means no human-provided labels are needed — the surrounding words in the corpus provide the training signal automatically. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Maximize Data Likelihood
For a corpus of length \(T\) and context window size \(m\), the Skip-Gram model maximizes:

\[L(\theta) = \prod_{t=1}^{T} \prod_{\substack{-m \leq j \leq m \\ j \neq 0}} P(w_{t+j} \mid w_t;\, \theta)\]

Where:

- \(\theta\) is the set of all model parameters
- \(m\) is the window size
- \(w_t\) is the center word
- \(t = 1, \ldots, T\) indexes positions in the corpus

### Minimize Cost Function
Instead of maximizing likelihood directly, we minimize the average negative log-likelihood:

\[J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \leq j \leq m \\ j \neq 0}} \log P(w_{t+j} \mid w_t;\, \theta)\]

> ⬆ **ADDED — Why negative log-likelihood?** Products over large corpora are numerically unstable (very small floats). Taking the log converts products into sums, which are stable and easier to differentiate. Negating it turns the maximization problem into a standard minimization. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Embedding Similarity

### Dot Product as Similarity
The probability \(P(w_{t+j} \mid w_t)\) should be high when the center word and context word are likely to co-occur.

Two embeddings are considered similar when their dot product is large:

\[\text{Similarity}(w_{t+j}, w_t) \approx \mathbf{w}_{t+j}^T \mathbf{w}_t\]

> ⬆ **ADDED:** A larger dot product means the two vectors point in more similar directions in embedding space. This is the raw similarity score before it is converted to a probability by softmax. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Two Vectors Per Word

### Center and Context Embeddings
In Skip-Gram, each word has two learned vectors:

- One vector when the word acts as a center word
- One vector when the word acts as a context word

For a center word \(c\) and context word \(o\), we define \(P(o \mid c)\) using their similarity.

> ⬆ **ADDED — Why two vectors?** Keeping separate embeddings for "center" and "context" roles avoids a word being similar to itself by construction, which would otherwise inflate the softmax probability for the target word appearing in its own context window. In practice, after training, you typically use only the center-word vectors \(\mathbf{v}\) for downstream tasks, discarding the context vectors \(\mathbf{u}\). [cs.umd](https://www.cs.umd.edu/class/fall2019/cmsc470/slides/slides_10.pdf)

***

## Softmax Formulation

### Context Probability
For a center word \(c\) and context word \(o\):

\[P(o \mid c) = \frac{\exp(\mathbf{u}_o^T \mathbf{v}_c)}{\sum_{w \in V} \exp(\mathbf{u}_w^T \mathbf{v}_c)}\]

Where:

- \(\mathbf{v}_c\) is the center-word embedding
- \(\mathbf{u}_o\) is the context-word embedding
- \(V\) is the vocabulary

This softmax turns similarity scores into a probability distribution.

> ⬆ **ADDED — The computational problem:** The denominator sums \(\exp(\mathbf{u}_w^T \mathbf{v}_c)\) over **every word in the vocabulary** on every training step. With \(|V| \approx\) tens of thousands, this is extremely slow — this is the direct motivation for **negative sampling** below.  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Model Parameters

### Parameter Set
If the embedding dimension is \(d\) and the vocabulary size is \(|V|\), then the model learns two embeddings per word, for a total of \(2d|V|\) parameters.

A schematic view of the parameters is:

\[\theta = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \\ \vdots \\ \mathbf{v}_{|V|} \\ \mathbf{u}_1 \\ \mathbf{u}_2 \\ \vdots \\ \mathbf{u}_{|V|} \end{bmatrix}\]

These parameters are learned using stochastic gradient descent.

> ⬆ **ADDED — Quiz computation example:** If \(|V| = 10{,}000\) and \(d = 300\), the model learns \(2 \times 300 \times 10{,}000 = 6{,}000{,}000\) parameters.  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Negative Sampling

### Motivation
The softmax denominator sums over the entire vocabulary, which is computationally expensive.

Instead of comparing a target word against every possible negative context word, negative sampling selects only a small number of negative examples.

> ⬆ **ADDED — Key distinction:** Negative sampling **replaces the full softmax objective** with a binary classification task: distinguish true context word pairs (positive) from randomly sampled noise pairs (negative). This makes each training step much cheaper while still producing useful embeddings. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Negative Sampling Training Example

### Example Window
Consider the sentence fragment:

`... lemon, a [tablespoon of apricot jam, a] pinch ...`

If the target word is `apricot` and the context window is \(L = \pm 2\), then the context words are:

- tablespoon
- of
- jam
- a

This creates positive training pairs:

- \((\text{apricot}, \text{tablespoon})\)
- \((\text{apricot}, \text{of})\)
- \((\text{apricot}, \text{jam})\)
- \((\text{apricot}, \text{a})\)

Negative examples are formed by pairing the target with random noise words, such as:

- \((\text{apricot}, \text{aardvark})\)
- \((\text{apricot}, \text{seven})\)
- \((\text{apricot}, \text{forever})\)
- \((\text{apricot}, \text{coaxial})\)

> ⬆ **ADDED:** The number of negative samples per positive pair is the hyperparameter \(k\) (introduced below). For each positive pair, you draw \(k\) random words from the vocabulary to serve as negatives. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Negative Sampling Objective

### Goal
The objective is to:

- maximize similarity between the target word and true context words
- minimize similarity between the target word and sampled negative words

For one positive pair \((w, c_{\text{pos}})\) and \(k\) negative samples \(c_{\text{neg}_1}, \ldots, c_{\text{neg}_k}\), the loss is:

\[L_{CE} = -\log \left( P(+ \mid w, c_{\text{pos}}) \prod_{i=1}^{k} P(- \mid w, c_{\text{neg}_i}) \right)\]

Expanding the log gives:

\[L_{CE} = -\left( \log P(+ \mid w, c_{\text{pos}}) + \sum_{i=1}^{k} \log P(- \mid w, c_{\text{neg}_i}) \right)\]

Since \(P(- \mid w, c_{\text{neg}_i}) = 1 - P(+ \mid w, c_{\text{neg}_i})\), we can write:

\[L_{CE} = -\left( \log P(+ \mid w, c_{\text{pos}}) + \sum_{i=1}^{k} \log \left( 1 - P(+ \mid w, c_{\text{neg}_i}) \right) \right)\]

Using the sigmoid function \(\sigma(\cdot)\), the final form is:

\[L_{CE} = -\left( \log \sigma(c_{\text{pos}} \cdot w) + \sum_{i=1}^{k} \log \sigma(-c_{\text{neg}_i} \cdot w) \right)\]

Where:

- \(k\) is a hyperparameter
- Common values for \(k\) are 3 to 10

> ⬆ **ADDED — Why sigmoid?** \(P(+ \mid w, c) = \sigma(c \cdot w)\) maps the dot-product similarity score to a probability between 0 and 1. For a negative sample, we want \(P(+ \mid w, c_{\text{neg}})\) to be low, so \(\sigma(-c_{\text{neg}} \cdot w)\) flips the sign — making a large dot product produce a *low* positive probability, i.e., a high negative probability. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Quiz trap:** \(\log \sigma(-c_{\text{neg}} \cdot w)\) is **not** the same as \(-\log \sigma(c_{\text{neg}} \cdot w)\). The minus sign is **inside** the sigmoid argument, not outside the log. This is a common point of confusion.

***

## Optimization Objective

### Intuition
Training adjusts the embeddings so that:

- positive pairs become more likely
- negative pairs become less likely
- this happens across the entire training set

Intuitively:

- move `apricot` closer to `jam`
- move `apricot` farther from unrelated words like `matrix` or `Tolstoy`

> ⬆ **ADDED:** After many gradient updates across the full corpus, the geometry of the embedding space reflects semantic and syntactic structure — words that appear in similar contexts end up with similar vectors. This is the distributional hypothesis encoded into a continuous space. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Derivatives of the Loss Function

### Gradient With Respect to the Positive Context Vector
\[\frac{\partial L_{CE}}{\partial c_{\text{pos}}} = \left[ \sigma(c_{\text{pos}} \cdot w) - 1 \right] w\]

### Gradient With Respect to a Negative Context Vector
\[\frac{\partial L_{CE}}{\partial c_{\text{neg}}} = \left[ \sigma(c_{\text{neg}} \cdot w) \right] w\]

### Gradient With Respect to the Target Word Vector
\[\frac{\partial L_{CE}}{\partial w} = \left[ \sigma(c_{\text{pos}} \cdot w) - 1 \right] c_{\text{pos}} + \sum_{i=1}^{k} \left[ \sigma(c_{\text{neg}_i} \cdot w) \right] c_{\text{neg}_i}\]

> ⬆ **ADDED — Conceptual interpretation of gradients (likely more testable than computation):**
> - The term \(\sigma(c_{\text{pos}} \cdot w) - 1\) is always **negative** (since \(\sigma \in (0,1)\)), so the positive context vector is pushed **toward** \(w\).
> - The term \(\sigma(c_{\text{neg}} \cdot w)\) is always **positive**, so negative context vectors are pushed **away** from \(w\).
> - This directly implements the optimization intuition: pull positives closer, push negatives farther. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## SGD Updates

### Update Rules
With learning rate \(\eta\), stochastic gradient descent updates the vectors as follows.

Positive context update:

\[c_{\text{pos}}^{(t+1)} = c_{\text{pos}}^{(t)} - \eta \left[ \sigma\left(c_{\text{pos}}^{(t)} \cdot w\right) - 1 \right] w\]

Negative context update:

\[c_{\text{neg}}^{(t+1)} = c_{\text{neg}}^{(t)} - \eta \left[ \sigma\left(c_{\text{neg}}^{(t)} \cdot w\right) \right] w\]

Target word update:

\[w^{(t+1)} = w^{(t)} - \eta \left( \left[ \sigma\left(c_{\text{pos}} \cdot w^{(t)}\right) - 1 \right] c_{\text{pos}} + \sum_{i=1}^{k} \left[ \sigma\left(c_{\text{neg}_i} \cdot w^{(t)}\right) \right] c_{\text{neg}_i} \right)\]

> ⬆ **ADDED — Reading the update rules intuitively:**
> - For the **positive context** update: \(\sigma(\cdot) - 1 < 0\), so subtracting a negative value **adds** to \(c_{\text{pos}}\) in the direction of \(w\) → vectors move **closer**.
> - For the **negative context** update: \(\sigma(\cdot) > 0\), so subtracting a positive value **subtracts** from \(c_{\text{neg}}\) in the direction of \(w\) → vectors move **farther apart**.
> - The **learning rate** \(\eta\) controls step size; if too large, updates overshoot; if too small, training is slow. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Key Takeaways

- Sparse word representations are high-dimensional and difficult to generalize from.
- Word embeddings are dense, low-dimensional, real-valued representations.
- CBOW predicts a word from its context.
- Skip-Gram predicts context words from a center word.
- Skip-Gram uses dot-product similarity and softmax probabilities.
- Negative sampling makes training more efficient by avoiding a full softmax over the vocabulary.
- Training moves related words closer together in embedding space and unrelated words farther apart.

> ⬆ **ADDED — Likely Quiz Questions for This Section:**
> 1. *How many total parameters does Skip-Gram learn, given \(|V|\) and \(d\)?* → \(2d|V|\)
> 2. *CBOW predicts ___ from ___; Skip-Gram predicts ___ from ___.* → center word from context; context words from center word
> 3. *Why is negative sampling used?* → to avoid computing full softmax over the entire vocabulary on each update
> 4. *True or False: Word2Vec embeddings are sparse.* → False
> 5. *True or False: A more complex TF-IDF variant is always better than a simpler one.* → False (mirrors Week 02 Q2 trap wording)
> 6. *What does the negative sign inside \(\sigma(-c_{\text{neg}} \cdot w)\) accomplish?* → It makes a high dot-product score produce a low positive-pair probability, i.e., pushes the negative context away

---

## Lecture Content: Vector Semantics and Embeddings

### Overview
This lecture reviews the role of representation in a text mining pipeline and introduces traditional distributional representations for capturing word meaning. It explains the distributional hypothesis, shows different ways to define context, and motivates the use of co-occurrence statistics such as PMI to measure semantic relatedness.

### Outline
- Traditional distributional representation
- Dense vector representations
- Interesting properties of word vectors
- Applying word embeddings

***

## Recap: Text Mining Pipeline

### Typical Pipeline
A typical text mining pipeline consists of three broad stages:

- Preprocessing
- Representation
- Model training

### Purpose of Each Stage
The goal of preprocessing is to make text better formatted. Representation makes text data machine-actionable, and model training uses those representations for the final downstream task.

> ⬆ **ADDED — Why this recap matters for Week 03:** This pipeline was introduced in Week 01 and applied in Week 02. Week 03 focuses entirely on the **Representation** stage — specifically, how to represent word *meaning*, not just word *occurrence*. Quizzes across both prior weeks have tested pipeline stage definitions, so knowing which stage embeddings belong to is a likely target. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/d6714559-d19f-40ff-8f12-3bbaab921922/week01_content.md)

***

## Vector Semantics

### Main Goal
The central goal of vector semantics is to preserve word meaning as much as possible when converting language into a numeric representation.

> ⬆ **ADDED — Contrast with Week 02:** Bag-of-words and TF-IDF also produce numeric representations, but they do **not** attempt to preserve meaning — they only record term occurrence statistics. Vector semantics is a step beyond: the goal is for the geometry of the vector space to reflect semantic relationships between words. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Traditional Distributional Representation

### Motivating Example: "tezguino"
The lecture uses the word `tezguino` as an example of how meaning can be inferred from context. From sentences such as "I will have a drink of tezguino", "Tezguino makes you drunk", and "We make tezguino out of corn, but we do not distill it", one can infer that `tezguino` is a kind of beer. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Core Idea
This example illustrates that a word's meaning can often be recovered from the distribution of words that appear near it.

> ⬆ **ADDED — Why this example is instructive for quizzes:** The `tezguino` example is the lecture's motivating case for the distributional hypothesis (below). If a quiz asks *"what is the basis for inferring word meaning in distributional models?"*, the answer is: **the surrounding context words** — not the word's spelling, etymology, or dictionary definition. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Word Embedding Idea

### Latent Semantic Space
The goal of a word embedding approach is to learn a latent space from a corpus that preserves semantic information as much as possible. Instead of representing words with sparse symbolic counts alone, we seek a lower-dimensional space that captures meaningful relationships between words. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — "Latent" means:** The dimensions of the embedding space are **not** directly interpretable as specific words or features. Unlike a TF-IDF vector where dimension 42 might mean "the word *apricot*", an embedding dimension has no fixed human-readable label. The structure emerges from the training process. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Distributional Hypothesis

### Basic Principle
The lecture presents John Firth's 1957 idea: "You shall know a word by the company it keeps." Words that often occur with similar context words tend to have similar meanings. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — This is a named, citable principle — high quiz priority.** Know the author (Firth), the year (1957), and the exact quote. Both CBOW and Skip-Gram are direct computational implementations of this hypothesis. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Similarity Is Not Synonymy
Distributional similarity does not necessarily mean exact synonymy. For example, pairs such as `dog` and `cat`, or `coffee` and `tea`, are semantically similar because they occur in related contexts, even though they are not identical in meaning. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Quiz trap:** A question may state: *"If two words are distributionally similar, they are synonyms."* This is **False**. `dog` and `cat` are distributionally similar (both appear near "pet", "feed", "walk") but are not synonyms. Distributional similarity captures *semantic relatedness*, not *identity of meaning*. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Extended examples of similar-but-not-synonymous pairs:**
> - `good` / `bad` — antonyms, but occur in very similar contexts → high embedding similarity (confirmed in Week 03 code: `wv.similarity('good', 'bad') = 0.719`) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - `king` / `queen` — related but not synonymous
> - This is an important limitation: distributional models cannot inherently distinguish antonyms from synonyms based on context alone.

***

## Defining Context

### Option 1: Document-Level Context
One way to define context is to treat the entire document as the context in which a word appears. This leads to a term-document matrix whose dimensions are roughly \(|V| \times d\), where \(|V|\) is the vocabulary size and \(d\) is the number of documents or features.  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Properties of the Term-Document Matrix
In this setup, the vocabulary size may be on the order of 10K to 50K frequent words, and the document dimension can be even larger. The resulting vectors are high-dimensional and often sparse. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Raw Counts and TF-IDF
Raw count representations are skewed toward very common but less informative words such as `the` and `it`. TF-IDF can be viewed as a special case of this term-document style of distributional representation that reduces the dominance of extremely frequent words. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — TF-IDF connection back to Week 02:** This is the explicit bridge between Week 02 (TF-IDF) and Week 03 (embeddings). TF-IDF is a *document-level distributional representation* — it is a smarter version of raw counts, but it is still sparse and still indexed by vocabulary terms. Word2Vec replaces this entire paradigm with a *dense, local co-occurrence* approach. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/a311cb5c-3f50-4522-9108-876314359d2d/week02_content.md)

***

## Option 2: Nearby Co-occurrence

### Sliding Window Context
A second way to define context is by nearby words within a sliding window. In this formulation, context is based on local co-occurrence rather than full documents. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — How the window works:** For a window size of \(\pm 2\) around the word `apricot` in "tablespoon of **apricot** jam a", the context words are `tablespoon`, `of`, `jam`, `a`. This is the same window concept used in Skip-Gram training. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Term-Term Matrix
This approach leads to a term-term matrix, where rows correspond to target words and columns correspond to context words. Each entry records how strongly a target word co-occurs with a context word. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Term-Document vs. Term-Term Quick Reference:**

| Property | Term-Document Matrix | Term-Term Matrix |
|---|---|---|
| **Rows** | Vocabulary words | Vocabulary words |
| **Columns** | Documents | Context words (vocabulary) |
| **Entry meaning** | How often word appears in document | How often word co-occurs with context word |
| **Context scope** | Entire document | Local sliding window |
| **Size** | \(\|V\| \times \|D\|\) | \(\|V\| \times \|V\|\) |
| **Related method** | TF-IDF | PMI-weighted co-occurrence |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Sparsity Problem
Even with local co-occurrence, the resulting vectors still tend to be sparse. Raw counts are again skewed toward frequent non-informative words such as `the` and `it`, which makes direct count-based representations less effective without further weighting. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Why sparsity persists:** Even with a sliding window, most word pairs in a large vocabulary never co-occur. A term-term matrix of size \(|V| \times |V|\) with \(|V| = 50{,}000\) would have 2.5 billion entries — the vast majority of which would be zero. This is exactly why PMI (below) and later dense embeddings (Word2Vec) were developed.  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Measuring Co-occurrence Strength

### Why Raw Frequency Is Not Enough
A word pair that appears frequently is not always the most informative pair. Some co-occurrences are common simply because one or both words are very frequent, not because the pair carries strong semantic association. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Connecting to Zipf's Law (Week 01):** This is the same problem introduced with stopwords in Week 01. Words like `the`, `it`, and `of` appear with nearly everything — so raw co-occurrence with those words tells you almost nothing about semantic meaning. PMI corrects for this by asking: does this pair co-occur *more than you'd expect by chance*? [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/d6714559-d19f-40ff-8f12-3bbaab921922/week01_content.md)

### Pointwise Mutual Information
To address this issue, the lecture introduces Pointwise Mutual Information, or PMI, as a way to measure the strength of association between a target word and a context word. PMI rewards word pairs that co-occur more often than expected by chance. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — PMI Formula** *(not in original notes — high quiz risk given Week 02's formula computation questions)*:

\[\text{PMI}(w, c) = \log_2 \frac{P(w, c)}{P(w) \cdot P(c)}\]

Where:
- \(P(w, c)\) = probability that word \(w\) and context word \(c\) co-occur
- \(P(w)\) = marginal probability of word \(w\)
- \(P(c)\) = marginal probability of context word \(c\)

**Interpretation:**
- \(\text{PMI} > 0\): the pair co-occurs **more** than expected by chance → informative association
- \(\text{PMI} = 0\): co-occurrence is exactly what chance predicts → no association
- \(\text{PMI} < 0\): co-occurs **less** than expected → avoidance (often treated as 0 in practice, called **Positive PMI / PPMI**)

> ⬆ **ADDED — Quiz trap:** PMI can be negative for pairs that rarely co-occur. In practice, **PPMI (Positive PMI)** replaces negative values with 0, since negative PMI values are unreliable with sparse data. If the quiz asks about PMI limitations, sparse counts and negative values are the answer. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## PMI vs. Frequency

### Example Comparisons
The lecture compares frequency and PMI for word pairs involving `drink` under an `object of` relation. Although pairs such as `drink` with `bunch beer`, `tea`, and `liquid` each have frequency 2, their PMI values remain high, while the pair `drink` with `it` has frequency 3 but a much lower PMI of 1.25. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Interpretation
This shows that PMI can identify informative associations better than raw frequency alone. A frequent but generic word like `it` may co-occur often without being semantically useful, while rarer but more specific words can have much stronger association scores. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Memorization anchor for this example:**
> - `drink` + `it` → frequency **3**, PMI **1.25** → *frequent but semantically weak*
> - `drink` + `tea` → frequency **2**, PMI **high** → *less frequent but semantically strong*
> - The takeaway: **higher frequency ≠ higher semantic relevance**. PMI normalizes for the base rates of each word. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Key Takeaways

- Representation is the stage of the NLP pipeline that makes text machine-actionable.
- Traditional distributional methods infer meaning from word context.
- The distributional hypothesis states that words with similar contexts tend to have similar meanings.
- Context can be defined at the document level or through nearby co-occurrence.
- Term-document and term-term matrices are often sparse and biased toward common function words.
- PMI is a better measure of association than raw count when the goal is to capture semantic relatedness.

> ⬆ **ADDED — Likely Quiz Questions for This Section:**
> 1. *What is the distributional hypothesis?* → Words that appear in similar contexts tend to have similar meanings (Firth, 1957)
> 2. *True or False: Distributional similarity implies synonymy.* → **False** — `good` and `bad` are distributionally similar but are antonyms
> 3. *What is the difference between a term-document matrix and a term-term matrix?* → Term-document uses full documents as context; term-term uses a local sliding window
> 4. *Why is PMI preferred over raw co-occurrence frequency?* → PMI controls for the base rate frequency of each word; raw frequency is biased toward common function words
> 5. *What does PMI = 0 mean?* → The pair co-occurs exactly as often as expected by chance — no special association
> 6. *What is PPMI and why is it used?* → Positive PMI — replaces negative PMI values with 0 because negative values are unreliable with sparse data

---

## Lecture Content: Interesting Properties and Applying Word Embeddings

### Overview
This lecture covers two major topics: the linguistic regularities captured by learned word vectors and practical ways to apply embeddings in NLP tasks. It highlights analogy-style behavior in embedding space and introduces common pre-trained embeddings as well as reasons to train embeddings on domain-specific corpora.

### Outline
- Traditional distributional representation
- Dense vector representations
- Interesting properties of word vectors
- Applying word embeddings

***

## Interesting Properties of Word Vectors

### Main Idea
Word vectors capture many linguistic regularities. Learned embeddings often organize semantic and syntactic relationships in a way that can be expressed through simple vector operations. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Why "regularities" matters conceptually:** This is a key distinguishing property of dense embeddings vs. sparse representations. A TF-IDF vector for `king` and a TF-IDF vector for `queen` have no predictable geometric relationship to each other. In a Word2Vec space, the *offset* between them is approximately the same as the offset between `man` and `woman`. This emergent structure is not explicitly programmed — it arises from training on co-occurrence patterns. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### Famous Analogy Example
A classic example is:

\[\text{vector}(\text{king}) - \text{vector}(\text{man}) + \text{vector}(\text{woman}) \approx \text{vector}(\text{queen})\]

This example illustrates that certain relationships can appear as approximately consistent vector offsets in embedding space. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — How to read this formula:** Subtracting `man` from `king` isolates the concept of royalty without maleness. Adding `woman` re-applies gender in the female direction. The result is a vector close to `queen` in embedding space. This is often phrased as: *"king is to man as queen is to woman."* [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Confirmed by Week 03 code notebook:**
> ```python
> woman = wv['queen'] - wv['king'] + wv['man']
> real_woman = wv['woman']
> cosine_similarity(woman, real_woman) = 0.7187
> ```
> The reconstructed vector is ~72% similar to the actual `woman` vector — not perfect, but structurally meaningful. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Quiz trap:** The analogy result is **approximate**, not exact. The formula produces a vector *near* `queen`, not the `queen` vector itself. Embeddings capture statistical regularities — they are not a perfect symbolic knowledge base.

### Why This Matters
This behavior suggests that embeddings do more than memorize co-occurrence counts. They capture structured relationships between words that can reflect gender, tense, plurality, and other semantic or syntactic patterns. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Additional relationship types captured by embeddings:**

| Relationship Type | Example Analogy |
|---|---|
| Gender | `king` – `man` + `woman` ≈ `queen` |
| Plural | `car` – `cars` + `cities` ≈ `city` |
| Tense | `walking` – `walked` + `swam` ≈ `swimming` |
| Geography | `France` – `Paris` + `Berlin` ≈ `Germany` |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Limitation:** Not all analogies work cleanly. Embedding spaces are statistical — they reflect patterns in the training corpus, including biases. For example, gender analogies can encode societal stereotypes present in the training data. This is a known and studied limitation of distributional embeddings.

***

## Applying Word Embeddings

### Using Pre-trained Embeddings
A common practical approach is to use pre-trained word embeddings. These embeddings are trained once on very large corpora and then reused in downstream tasks. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Why reuse pre-trained embeddings instead of training from scratch?**
> - Training requires a very large corpus and significant compute
> - Pre-trained models (e.g., Google News Word2Vec) already encode broad semantic knowledge from billions of words
> - Reusing them is faster and often produces better results on small datasets where training from scratch would overfit or underfit [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Popular Pre-trained Embeddings

### Google News Word2Vec
The lecture lists `GoogleNews-vectors-negative300.bin.gz` as a popular pre-trained embedding set.

- 300 dimensions
- 3 million words and phrases
- Trained on news text with roughly 100 billion words

> ⬆ **ADDED — Loaded in Week 03 code as:**
> ```python
> wv = api.load('word2vec-google-news-300')
> ```
> Each word vector is a `numpy.ndarray` of length 300. Words not in the vocabulary raise a `KeyError`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### GloVe 6B
Another popular option is `GloVe 6B`.

- Multiple dimensionality choices, such as 100 dimensions
- About 400K uncased vocabulary entries
- Trained on Wikipedia 2014 and Gigaword 5
- About 6 billion tokens

> ⬆ **ADDED — GloVe vs. Word2Vec Quick Reference:**

| Property | Word2Vec (Google News) | GloVe 6B |
|---|---|---|
| **Dimensions** | 300 | 50, 100, 200, or 300 |
| **Vocabulary** | 3 million words & phrases | ~400K uncased words |
| **Training data** | Google News (~100B words) | Wikipedia + Gigaword (~6B tokens) |
| **Training method** | Skip-Gram + negative sampling | Global co-occurrence matrix factorization |
| **Case sensitive** | Yes (`king` ≠ `King`) | No (uncased) |
| **Python loading** | `gensim` API | `KeyedVectors` after `glove2word2vec` conversion |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Quiz trap:** Word2Vec and GloVe both produce dense word embeddings, but they use **different training methods**. Word2Vec uses a neural predictive model (Skip-Gram / CBOW). GloVe uses matrix factorization over a global co-occurrence matrix. Both produce similar-quality embeddings in practice.

> ⬆ **ADDED — Confirmed by Week 03 code — similarity scores are close but not identical across models:**
> ```
> 'car'  'minivan'  GloVe: 0.67  Word2Vec: 0.69
> 'car'  'bicycle'  GloVe: 0.69  Word2Vec: 0.54
> 'car'  'airplane' GloVe: 0.65  Word2Vec: 0.42
> 'car'  'cereal'   GloVe: 0.12  Word2Vec: 0.14
> ```
> Both models agree that `cereal` is unrelated to `car`, but differ on how similar `bicycle` and `airplane` are — reflecting differences in training data and method. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Why Pre-trained Embeddings Help

### Practical Benefit
Pre-trained embeddings are useful because they already encode broad semantic structure learned from very large corpora. In many applications, they provide better similarity behavior and stronger initialization than training from scratch on a small dataset. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — "Stronger initialization" explained:** When using embeddings as input features for a downstream model (e.g., a classifier), starting from pre-trained vectors means the model already has a meaningful geometric structure to work with, rather than random noise. This typically speeds up training and improves final performance, especially with limited labeled data. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Train Your Own Embedding

### When to Train a Custom Model
For domain-specific applications, embeddings trained on in-domain text can be especially helpful. If your vocabulary, terminology, or style differs from general corpora, a custom embedding model may better capture relevant semantic relationships. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Concrete example of when custom training matters:** A medical corpus contains words like `tachycardia`, `metformin`, or `bradykinesia` that are absent from or poorly represented in Google News. Pre-trained general embeddings would have no vector for these terms or would embed them inaccurately. Training on clinical notes directly captures how these terms are used in context. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

### What You Need
To train your own embeddings, the main requirement is a sufficiently large raw corpus from the target domain. You can also initialize training from pre-trained vectors instead of starting completely from random values. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Quiz trap:** Training your own embeddings does **not** require labeled data. Word2Vec training is **self-supervised** — the supervision signal comes from the corpus structure itself (surrounding words), not from human-assigned labels. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Common Python Tools

### Libraries Mentioned
The lecture mentions several common tools for working with embeddings:

- `gensim`
- `fasttext`

These tools are widely used for training, loading, and querying embedding models in Python. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — What each tool does:**

| Tool | Primary Use | Notable Feature |
|---|---|---|
| `gensim` | Load, train, and query Word2Vec / GloVe models | `wv.most_similar()`, `wv.similarity()`, `api.load()` |
| `fasttext` | Train embeddings with subword information | Handles out-of-vocabulary words by decomposing into character n-grams |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Key gensim operations from the Week 03 code notebook (likely quiz targets given Week 01's pattern of testing specific function names):**

```python
wv.similarity('car', 'truck')          # cosine similarity between two words → float
wv.most_similar(positive=['car'], topn=20)  # nearest neighbors in embedding space
wv.doesnt_match(['fire','water','land','car'])  # find the outlier word
```

- `wv.similarity()` returns cosine similarity between two word vectors [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- `wv.most_similar()` returns the top-N nearest neighbors by cosine similarity [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- `wv.doesnt_match()` finds the word whose vector is most distant from the group centroid [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Cosine similarity formula** *(appeared in Week 03 code; formula computation questions appeared in Week 02)*:

\[\text{cosine\_similarity}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}\]

Where \(\|a\|\) is the L2 norm of vector \(a\). Values range from -1 (opposite) to +1 (identical direction).  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — doc2vec via averaging (from Week 03 code notebook):** When you need a single vector to represent a whole document, one simple approach is to **average** the word vectors of all tokens in the document:
> ```python
> def doc2vec(doc, wv):
>     vecs = [wv[token] for token in doc.split() if token in wv]
>     return np.mean(vecs, axis=0)
> ```
> This is fast and practical but loses word order. Note the counterintuitive result from the notebook:
> ```
> cos_sim("I like this course so much", "I do not like this course") = 0.931
> ```
> Despite opposite sentiment, the averaged vectors are nearly identical — because averaging washes out the effect of negation (`not`). This directly connects back to the **bag-of-words limitation** from Week 02. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Coding Focus

### Hands-on Usage
The lecture ends by transitioning into coding practice. This reflects an important point: embeddings are most useful when you actually load them, inspect nearest neighbors, compare similarities, and test them on downstream NLP tasks. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Code outputs worth memorizing for the quiz (specific numbers appeared in Week 01 Q1 as a tested output):**
> - `wv.similarity('good', 'bad') = 0.719` — antonyms are similar in embedding space [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - `wv.similarity('king', 'queen') = 0.65` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - `wv.doesnt_match(['fire','water','land','sea','air','car']) = 'car'` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

## Key Takeaways

- Word vectors can capture meaningful semantic and syntactic regularities.
- Analogy behavior emerges from geometric structure in embedding space.
- Pre-trained embeddings are useful when you want strong general-purpose semantic representations.
- Domain-specific corpora can justify training custom embeddings.
- Libraries such as `gensim` and `fasttext` make embedding workflows practical in Python.

> ⬆ **ADDED — Likely Quiz Questions for This Section:**
> 1. *What does `vector(king) - vector(man) + vector(woman)` approximate?* → `vector(queen)`
> 2. *True or False: Pre-trained embeddings are always better than custom-trained embeddings.* → **False** — for domain-specific vocabulary, custom training can be better
> 3. *What Python function returns the nearest neighbor words to a given word in gensim?* → `wv.most_similar()`
> 4. *What similarity metric does `wv.similarity()` compute?* → cosine similarity
> 5. *Why might averaging word vectors fail for sentiment tasks?* → Averaging washes out negation (e.g., "not good" ≈ "very good" after averaging), losing the effect of individual important words
> 6. *Word2Vec and GloVe both produce dense embeddings — what is the key difference between them?* → Word2Vec uses a predictive (neural) training objective; GloVe uses global co-occurrence matrix factorization
> 7. *True or False: Training Word2Vec requires labeled data.* → **False** — it is self-supervised; labels come from the corpus structure itself

---

## Code Content: word2vec notebook

***

### Download a Word2Vec model pre-trained by Google

- Trained on part of Google News dataset (about 100 billion words)
- 300-dimensional vectors for 3 million words and phrases
- More details: https://code.google.com/archive/p/word2vec/

> ⬆ **ADDED — Quiz-ready facts about this model:**
> - Vocabulary: **3 million** words and phrases [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - Dimensions: **300** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - Training corpus: ~**100 billion** words from Google News [huggingface](https://huggingface.co/fse/word2vec-google-news-300)
> - Training method: **Skip-Gram with negative sampling** (the `negative300` in the filename refers to this) [huggingface](https://huggingface.co/fse/word2vec-google-news-300)
> - Case-sensitive: `king` and `King` are **different vocabulary entries** with different vectors [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

```python
!pip install gensim
import gensim.downloader as api
# the download could take you some time
wv = api.load('word2vec-google-news-300')
vec_king = wv['king']
print(len(vec_king), vec_king)
print(type(vec_king))
```

```
300 [ 1.25976562e-01  2.97851562e-02  8.60595703e-03  1.39648438e-01
 -2.56347656e-02 -3.61328125e-02 ...] 
<class 'numpy.ndarray'>
```

> ⬆ **ADDED — What to know about this output:**
> - `len(vec_king) = 300` confirms the embedding dimension [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - `type(vec_king)` is `numpy.ndarray` — each word vector is a plain NumPy array, not a special gensim object [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - Entries are `float32` values, both positive and negative, consistent with dense embeddings (contrast with TF-IDF which is always ≥ 0) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

```python
try:
    vec_cameroon = wv['cameroon']
except KeyError:
    print("The word 'cameroon' does not appear in this model")
```

```
The word 'cameroon' does not appear in this model
```

> ⬆ **ADDED — Key behavior: out-of-vocabulary (OOV) words raise `KeyError`** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - Accessing a word not in the vocabulary raises `KeyError`, not a zero vector or `None`
> - The correct pattern is to wrap lookups in a `try/except KeyError` block
> - This is exactly what `doc2vec()` does below with the `except KeyError: pass` pattern
> - **Quiz trap:** The model has 3 million entries, but that does not mean every word is covered. Rare words, misspellings, and many proper nouns may be absent.

***

```python
# has the upper words in the vocab
wv['King'] 
```

```
array([-0.00350952,  0.01623535, -0.08154297,..., 0.24902344],dtype=float32)
```

> ⬆ **ADDED — `king` vs. `King` are different entries:**
> ```
> wv.similarity('king', 'King') = 0.52
> ```
> Despite referring to the same concept, the lowercase and capitalized forms have different vectors because they appeared in different syntactic positions in the training corpus (e.g., `King` often begins sentences or refers to proper names). This contrasts with `CountVectorizer`'s default behavior in Week 02, which **lowercases** everything. Word2Vec preserves case. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/a311cb5c-3f50-4522-9108-876314359d2d/week02_content.md)

***

```python
pairs = [
    ('king', 'King'),
    ('car', 'truck'),
    ('king', 'queen'),
    ('UCSD', 'Rady'),
    ('UCSD', 'Scripps'),
    ('UCSD', 'UCLA'),
    ('Scripps', 'Rady'),
    ('sports', 'football'),
    ('basketball', 'football'),
    ('good', 'bad'),
]
for w1, w2 in pairs:
    sim = wv.similarity(w1, w2)
    print('%r\t%r\t%.2f' % (w1, w2, sim))
```

```
'king'         'King'         0.52
'car'          'truck'        0.67
'king'         'queen'        0.65
'UCSD'         'Rady'         0.35
'UCSD'         'Scripps'      0.39
'UCSD'         'UCLA'         0.62
'Scripps'      'Rady'         0.23
'sports'       'football'     0.59
'basketball'   'football'     0.67
'good'         'bad'          0.72
```

> ⬆ **ADDED — Annotated interpretation of key results (likely quiz material given Week 01's pattern of testing exact output):**

| Pair | Score | Why It's Interesting |
|---|---|---|
| `king` / `King` | 0.52 | Same concept, different case → moderate similarity only; model is case-sensitive |
| `king` / `queen` | 0.65 | Semantically related (royalty), captures gender relationship |
| `good` / `bad` | **0.72** | Antonyms score *higher* than `king`/`queen` — distributional similarity ≠ synonymy |
| `UCSD` / `UCLA` | 0.62 | Both universities → similar contexts in corpus |
| `basketball` / `football` | 0.67 | Both sports → high similarity |
| `Scripps` / `Rady` | 0.23 | Both UCSD schools, but rarer co-occurrence → lower similarity |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — `good`/`bad` = 0.72 is the single most important result to remember:** It directly demonstrates that antonyms can have *higher* similarity scores than semantically related words, because they appear in identical contexts ("this is *good*" / "this is *bad*"). This is the empirical proof of the "similarity ≠ synonymy" point from the distributional hypothesis. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

```python
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(word_a, word_b, wv):
    try:
        a = wv[word_a]
        b = wv[word_b]
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        return cos_sim
    except KeyError:
        return -1

print(cosine_similarity('car', 'minivan', wv))
```

```
0.69070363
```

> ⬆ **ADDED — Know this formula cold** *(Week 02 had computation questions; this formula is directly implemented in the notebook)*:

\[\text{cosine\_similarity}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}\]

- `dot(a, b)` = dot product of the two vectors [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- `norm(a)` = L2 norm = \(\sqrt{\sum_i a_i^2}\) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- Returns a value between **-1** (opposite directions) and **+1** (identical directions) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
- Returns **-1** if either word is out of vocabulary (`KeyError`) — note this is a code convention, not a real similarity value [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — `wv.similarity()` vs. manual `cosine_similarity()`:** Both compute cosine similarity. `wv.similarity()` is gensim's built-in shortcut; the manual implementation uses `numpy.dot` and `numpy.linalg.norm`. Know both. [radimrehurek](https://radimrehurek.com/gensim/models/word2vec.html)

***

```python
diff1 = wv['queen'] - wv['king']
diff2 = wv['woman'] - wv['man']
diff3 = wv['moon'] - wv['sun']
print( dot(diff1, diff2)/(norm(diff1)*norm(diff2)) )
print( dot(diff1, diff3)/(norm(diff1)*norm(diff3)) )
```

```
0.4493644
-0.06825185
```

> ⬆ **ADDED — What this code tests:**
> - `diff1 = queen - king` captures the gender-royalty offset vector [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - `diff2 = woman - man` captures the generic gender offset vector [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - Their cosine similarity is **0.45** — reasonably high, confirming the analogy structure is real but not perfect [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - `diff3 = moon - sun` is an unrelated offset → cosine similarity ≈ **-0.07** ≈ 0, meaning no relationship [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - **Quiz interpretation:** If two word-pair differences have high cosine similarity, the pairs encode a similar relationship in embedding space. Near-zero cosine similarity means no shared structure.

***

```python
woman = wv['queen'] - wv['king'] + wv['man']
real_woman = wv['woman']
print( dot(woman, real_woman)/(norm(woman)*norm(real_woman)) )
```

```
0.7186802
```

> ⬆ **ADDED — This is the direct computational test of the analogy formula:**
> \[\text{vector}(\text{king}) - \text{vector}(\text{man}) + \text{vector}(\text{woman}) \approx \text{vector}(\text{queen})\]
> The reconstructed vector is ~**72% similar** to the actual `woman` vector — strong evidence the analogy holds, but not exact. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Quiz trap:** The analogy produces a vector *near* the target, not the exact target vector. This is expected — embeddings encode statistical regularities, not symbolic rules.

***

### Nearest Neighbor Search

Most similar words based on word embedding vectors.

```python
print(wv.most_similar(positive=['car'], topn=20))
```

```
[('vehicle', 0.782), ('cars', 0.742), ('SUV', 0.716), ('minivan', 0.691),
 ('truck', 0.674), ('Car', 0.668), ('Ford_Focus', 0.667), ...]
```

> ⬆ **ADDED — What `most_similar()` does:** Returns the top-N words with the highest cosine similarity to the query word(s). The result is a ranked list of `(word, similarity_score)` tuples. [radimrehurek](https://radimrehurek.com/gensim/models/word2vec.html)

> ⬆ **ADDED — Notable observations from the `car` neighbors:**
> - `vehicle` (0.782) is the closest — a true hypernym (broader category)
> - `cars` (0.742) is the plural — morphological variant
> - `Car` (0.668) is the capitalized form — case-sensitive; treated as a separate word
> - Specific car models appear (`Ford_Focus`, `Honda_Civic`, `Toyota_Camry`) — phrases in the vocabulary are represented with underscores [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — `most_similar` with analogy queries:** `most_similar` also supports the analogy pattern directly:
> ```python
> wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
> # Returns: [('queen', ~0.71)]
> ```
> This is the standard gensim way to run analogy tasks without manual vector arithmetic. [radimrehurek](https://radimrehurek.com/gensim/models/word2vec.html)

***

### Find the outlier in a list of words

```python
print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))
```

```
car
```

> ⬆ **ADDED — How `doesnt_match()` works:** Computes the centroid (average vector) of all words in the list, then finds the word whose vector is most distant from that centroid by cosine similarity. `car` is the outlier because it belongs to a transportation semantic cluster while `fire`, `water`, `land`, `sea`, and `air` all belong to natural elements. [radimrehurek](https://radimrehurek.com/gensim/models/word2vec.html)

> ⬆ **ADDED — Quiz trap:** `doesnt_match()` uses the **mean vector** of the group, not pairwise comparisons. It finds the single word most unlike the group centroid — it does not guarantee it finds the "correct" answer by any external definition.

***

### Average word embedding for a doc

```python
def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

import numpy as np

def doc2vec(doc, wv):
    vecs = []
    for token in doc.split():
        try:
            vecs.append(wv[token])
        except KeyError:
            pass
    return np.mean(vecs, axis=0)

doc1 = 'I like this course so much'
doc2 = 'I do not like this course'
doc3 = 'I like this movie'
doc4 = 'I hate this movie'

v1 = doc2vec(doc1, wv)
v2 = doc2vec(doc2, wv)
v3 = doc2vec(doc3, wv)
v4 = doc2vec(doc4, wv)

print(cos_sim(v1, v2))
print(cos_sim(v1, v3))
print(cos_sim(v3, v4))
```

```
0.93147177
0.8114622
0.9014764
```

> ⬆ **ADDED — Annotated interpretation of the three results (high quiz risk — counterintuitive outputs):**

| Pair | Score | Why |
|---|---|---|
| `"I like this course so much"` vs `"I do not like this course"` | **0.931** | Nearly identical — averaging washes out `not`; the shared words `I`, `like`, `this`, `course` dominate |
| `"I like this course"` vs `"I like this movie"` | **0.811** | Same sentiment, different topic (`course` vs `movie`) — lower because the topic words differ |
| `"I like this movie"` vs `"I hate this movie"` | **0.901** | Opposite sentiment, same topic — averaging `hate` and `like` produces nearly identical document vectors |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — The critical lesson from these numbers:** Averaged word embeddings are **bad at capturing sentiment** because:
> 1. Negation words like `not` have low individual weight after averaging [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> 2. Antonyms like `like` and `hate` appear in the same contexts → similar vectors → nearly cancel each other out in the average [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> 3. **This connects directly back to Week 02:** bag-of-words also fails at negation. Averaged embeddings have the same weakness.

> ⬆ **ADDED — `doc2vec` implementation details to know:**
> - Tokenizes by `doc.split()` — simple whitespace split, **no preprocessing** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - OOV words are silently skipped (`except KeyError: pass`) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - Result is `np.mean(vecs, axis=0)` — element-wise average across all word vectors [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - **Quiz trap:** This `doc2vec` is a simple averaging baseline — it is **not** the Doc2Vec algorithm by Mikolov et al., which is a separate more sophisticated model.

***

```python
wv.similarity('good', 'bad')    # → 0.7190051
wv.similarity('hate', 'like')   # → 0.388815606
wv.similarity('long', 'short')  # → 0.57684326
```

> ⬆ **ADDED — These three results together reveal a pattern worth remembering:**

| Pair | Score | Relationship |
|---|---|---|
| `good` / `bad` | **0.719** | Antonyms — but score very high; both are general evaluative adjectives appearing in identical contexts |
| `long` / `short` | **0.577** | Antonyms — similarly high; dimensional adjectives share contexts |
| `hate` / `like` | **0.389** | Opposite sentiment verbs — notably *lower* than adjective antonyms; verbs appear in slightly more varied syntactic contexts |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

> ⬆ **ADDED — Quiz trap:** Do not assume all antonyms score equally high. The degree of contextual overlap determines the score. Adjective antonyms tend to score higher than verb antonyms because adjectives appear in more substitutable positions ("this is ___"). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

### Load GloVe and Compare

```python
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = './glove.6B/glove.6B.100d.txt'
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
```

> ⬆ **ADDED — GloVe loading workflow:**
> - GloVe vectors are distributed as plain `.txt` files, not in Word2Vec binary format [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - `glove2word2vec()` converts the GloVe text file into Word2Vec-compatible format [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - `KeyedVectors.load_word2vec_format()` then loads it into the same gensim interface as Word2Vec [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - After conversion, the `model` object supports all the same methods: `.similarity()`, `.most_similar()`, etc. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - The `DeprecationWarning` is expected — gensim now recommends loading GloVe directly via `KeyedVectors.load_word2vec_format(..., binary=False, no_header=True)` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)

***

```python
pairs = [
    ('car', 'minivan'),
    ('car', 'bicycle'),
    ('car', 'airplane'),
    ('car', 'cereal'),
    ('car', 'communism'),
]
for w1, w2 in pairs:
    sim_glove = model.similarity(w1, w2)
    sim_word2vec = wv.similarity(w1, w2)
    print('%r\t%r\t%.2f\t%.2f' % (w1, w2, sim_glove, sim_word2vec))
```

```
'car'  'minivan'    0.67    0.69
'car'  'bicycle'    0.69    0.54
'car'  'airplane'   0.65    0.42
'car'  'cereal'     0.12    0.14
'car'  'communism'  0.04    0.06
```

> ⬆ **ADDED — Side-by-side comparison analysis:**

| Pair | GloVe | Word2Vec | Takeaway |
|---|---|---|---|
| `car` / `minivan` | 0.67 | 0.69 | Agreement — close vehicle types |
| `car` / `bicycle` | **0.69** | 0.54 | GloVe scores higher — Wikipedia corpus captures broader vehicle relationships |
| `car` / `airplane` | **0.65** | 0.42 | GloVe scores much higher — global co-occurrence captures "transportation" broadly |
| `car` / `cereal` | 0.12 | 0.14 | Agreement — unrelated |
| `car` / `communism` | 0.04 | 0.06 | Agreement — completely unrelated |

 [datascientistinsights.substack](https://datascientistinsights.substack.com/p/glove-vs-word2vec-in-practice-does)

> ⬆ **ADDED — Why GloVe rates `bicycle` and `airplane` as more similar to `car` than Word2Vec does:** GloVe is trained on Wikipedia, which contains more encyclopedic text about transportation systems broadly. Word2Vec trained on news may associate `car` more narrowly with road vehicles. This reflects how training data domain affects embedding geometry. [datascientistinsights.substack](https://datascientistinsights.substack.com/p/glove-vs-word2vec-in-practice-does)

***

```python
model.similarity('better', 'worse')   # GloVe → 0.71766543
model.similarity('good', 'bad')       # GloVe → 0.7702799
```

> ⬆ **ADDED — Cross-model consistency check:**
> - GloVe: `good`/`bad` = **0.770**, Word2Vec: `good`/`bad` = **0.719** — both very high, both show antonym similarity [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - GloVe: `better`/`worse` = **0.718** — comparative antonyms also score high [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159433465/02047566-542f-42ef-84d6-79a90dc08feb/week03_content.md)
> - **Pattern:** Both models consistently show that evaluative antonym pairs score in the 0.70–0.77 range regardless of the embedding method, confirming this is a property of distributional learning, not a quirk of one specific model. [datascientistinsights.substack](https://datascientistinsights.substack.com/p/glove-vs-word2vec-in-practice-does)

***

> ⬆ **ADDED — Code Section: Likely Quiz Questions**
> 1. *What does `wv['word']` return, and what type is it?* → A 300-dimensional `numpy.ndarray` of float32 values
> 2. *What happens when you access a word not in the vocabulary?* → `KeyError` is raised
> 3. *Are `king` and `King` the same in Word2Vec?* → No — the model is case-sensitive; they have different vectors with similarity ~0.52
> 4. *What does `wv.most_similar(positive=['car'], topn=5)` return?* → The 5 words with highest cosine similarity to `car`, as a list of `(word, score)` tuples
> 5. *Why do `"I like this course"` and `"I do not like this course"` have 0.93 cosine similarity after doc2vec averaging?* → Averaging washes out `not`; the shared content words dominate the mean vector
> 6. *Why is `good`/`bad` similarity (~0.72) higher than `hate`/`like` similarity (~0.39)?* → Adjective antonyms appear in nearly identical syntactic positions and contexts; verb antonyms occur in more varied contexts
> 7. *What is the key difference in how GloVe and Word2Vec are loaded in gensim?* → Word2Vec loads directly via `api.load()`; GloVe requires conversion with `glove2word2vec()` first, then `KeyedVectors.load_word2vec_format()`

---
