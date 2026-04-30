# Week 04 
## Overview
* **Summary:**Language models are a core building block of both classic and modern NLP techniques. In this lecture, we will begin by introducing n-gram based probability distributions on natural language – how they are parameterized, learned, and evaluated.
* * After this week, you will be able to:
    * Define a language model and what it can be used for in practice
    * Describe the basics of n-gram-based language modeling techniques, along with their corresponding learning methods and smoothing techniques, and
    * Identify and apply the different ways in which language models are evaluated (e.g. held-out likelihood/perplexity) 

## Lecture Content
### 1.Types of Language Models
### 2.Introduction to N-Gram Language Models
### 3.Sparsity and N-Gram Language Models
### 4.N-Gram Estimation
### 5.Smoothing in N-Gram Language Models

---

## 1.Lecture Content: Types of Language Models
* **Summary:** A language model is a probability distribution over sequences of words or sentences. It assigns higher probability to better or more fluent word sequences. 

### Purpose
* Language models are used to:
    * Evaluate existing text.
    * Generate new text.

### Applications
* Speech recognition.
* Machine translation.
* Handwriting recognition.
* Optical character recognition.
* Document summarization.
* Dialog generation.
* Linguistic decipherment.
* Text style transfer.

* **Example:**In speech recognition, the acoustic model may allow several candidate transcriptions, and the language model prefers the most fluent sentence, such as "Night begins in one hour." over less plausible alternatives.

### *Good* Model
* **Summary:** A good language model gives high probability to good or fluent sentences. It can also generate text by sampling from the probability distribution.

### Possible Quiz Context
* A language model is not just a text generator.
* It also scores or evaluates text by assigning probabilities.
* Be able to distinguish n-gram, neural, and conditional language models.

---

## 2.Lecture Content: Introduction to N-Gram Language Models
* **Summary:**N-gram language models generate or score text from left to right using the chain rule, but simplify prediction by using only a limited recent context.

### Core Idea
* The full sentence probability can be decomposed with the chain rule.
* Conditioning on the entire left context is difficult.
* N-gram models solve this by making a Markov assumption.

### Markov Assumption
* **Summary:**The next word is predicted using only a limited number of previous words, not the full sentence history.

### Learning N-Gram Probabilities
* N-gram probabilities are estimated from corpus statistics.
* The model counts n-grams in a large training corpus.
* These estimates are maximum likelihood estimates.

### Example
* **Bigram-style example:**$P(door | the) = 0.0006$
* **Trigram-style example:**$P(door | close the) = 0.05$

### Increasing Order
* Higher-order n-grams capture more dependencies.
* Unigram: no previous context.
* Bigram: one previous word.
* Trigram: two previous words.
* Higher-order models often produce more locally coherent text.

### Possible Quiz Context
* The Markov assumption is an approximation, not a claim that earlier words never matter.
* Higher-order n-grams are usually more informative than lower-order n-grams.
* Higher-order n-grams also create more sparsity.

---

## 3.Lecture Content: Sparsity and N-Gram Language Models
* **Summary:**Sparsity means that many possible n-grams are unseen or very rare in the training corpus, which makes raw count-based estimates unreliable.

### Core Idea
* As n-gram order increases, the model becomes more specific.
* But the number of possible word sequences grows quickly.
* Many plausible sequences are never observed in training.

### Example
* The lecture contrasts observed phrases such as:
    * "please close the door"
    * "please close the window"
    * "please close the new"
    * "please close the gate"
* But a plausible continuation like:
    * "please close the first"
  may have count 0 in the training data.

### Why It Matters
* A zero count does not mean the phrase is impossible.
* It only means the phrase was not observed in the corpus.
* This causes raw maximum likelihood estimates to fail on unseen events.

### Higher vs Lower Order
* Higher-order n-grams are more specific but more sparse.
* Lower-order n-grams are denser and more general.

### Possible Quiz Context
* Sparsity gets worse as n-gram order increases.
* Zero count does not mean zero real-world plausibility.
* This sparsity problem is the reason smoothing is needed.

---

## 4.Lecture Content: N-Gram Estimation
* **Summary:**N-gram estimation uses observed counts to estimate next-word probabilities, but raw estimates from sparse data do not generalize well.

### Raw Estimation
* N-gram estimation begins with empirical counts from training data.
* These counts produce maximum likelihood estimates for conditional probabilities.

### Smoothing
* **Summary:**Smoothing flattens spiky distributions so they generalize better.

### Example
* For $P(w | denied the)$, the unsmoothed distribution heavily favors words seen in training, such as:
    * allegations
    * reports
    * claims
    * request
* A smoothed distribution reduces some of that mass and assigns probability to other unseen words.

### Evaluation
* Language model quality is evaluated using:
    * Test-set log likelihood
    * Perplexity

### Likelihood
* **Summary:**Log likelihood measures how much probability the model assigns to the test data.
* Higher likelihood is better.

### Perplexity
* **Summary:**Perplexity is the average per-word branching factor.
* Lower perplexity is better.

### Shannon's Game
* A language model can also be viewed through next-word prediction, as in:
    * "When I eat pizza, I wipe off the _____"

### Train, Validation, Test
* **Training data:**Used to estimate counts and parameters.
* **Held-out or validation data:**Used to tune hyperparameters such as smoothing choices.
* **Test data:**Used for final evaluation.

### Possible Quiz Context
* Smoothing redistributes probability mass, it does not just lower all probabilities.
* Higher likelihood is better, but lower perplexity is better.
* Do not tune hyperparameters on the test set.
* Held-out data is for tuning, test data is for final evaluation.

---

## 5.Lecture Content: Smoothing in N-Gram Language Models
* **Summary:**Smoothing methods address the tradeoff between specific but sparse higher-order n-grams and dense but general lower-order n-grams.

### Interpolation
* **Summary:**Linear interpolation mixes probabilities from multiple n-gram orders instead of relying on only one order.

### Why Interpolation Helps
* A 4-gram can be specific but sparse.
* A 2-gram can be more reliable but less specific.
* Interpolation combines both sources of information.

### Choosing Weights
* Interpolation weights can be chosen using:
    * Grid search
    * EM
    * Held-out data

### Possible Quiz Context
* Better interpolation methods smooth more when the context is less reliable.
* Interpolation mixes multiple orders rather than fully dropping to only one lower-order model.

### Discounting
* **Summary:**Discounting lowers observed counts because training counts tend to overestimate how often events will appear in future data.

### Absolute Discounting
* Reduce numerator counts by a constant $d$, for example $d = 0.75$.
* Redistribute the removed probability mass to unseen or less-supported events.

### Possible Quiz Context
* Absolute discounting does not erase seen events.
* It slightly lowers their counts so some probability mass can be reassigned.

### Fertility
* **Summary:**Context fertility is the number of distinct context types in which a word appears.

### Example
* The lecture motivates fertility with a prompt such as:
    * "There was an unexpected _____"
* A word like "delay" is more plausible in many contexts than a word like "Francisco" because fertility depends on contextual diversity, not just raw frequency.

### Kneser-Ney Smoothing
* **Summary:**Kneser-Ney smoothing combines discounting with a backoff model based on context fertility rather than raw frequency.

### Why Kneser-Ney Matters
* It is effective in practice.
* It uses the number of distinct contexts a word appears in.
* This makes it stronger than methods that rely only on frequency.

### What Actually Works
* Unigrams and bigrams are generally weak in practice.
* Trigrams are much better.
* 4-grams and 5-grams can be practically useful.
* Effective methods include:
    * Absolute discounting
    * Good-Turing
    * Held-out estimation
    * Witten-Bell
    * Kneser-Ney

### What N-Grams Capture
* N-grams capture many local correlations, including:
    * Word class restrictions
    * Morphology
    * Semantic class restrictions
    * Idioms
    * Some world knowledge
    * Some pop culture patterns

### Limitation
* N-grams do not capture long-distance dependencies well.

### Possible Quiz Context
* Frequency and fertility are not the same thing.
* Kneser-Ney is special because it uses context diversity, not just raw frequency.
* N-grams are good at local patterns but weak at long-distance dependencies.

---

