# Week06

## Overview
* POS tagging is one of the keystones in the NLP pipeline. It shows the machine what the syntactic role of every token is. In this lecture, we will show a series of POS tagging models from the most straightforward token classification perspective to the most recent sequence labeling methods built upon pre-trained neural language models.
* After this week, you will be able to:
    * Define/Describe the semantic meanings of the POS tags,
    * Describe the concepts of sequence labeling framework and conditional random field, and
    * Apply popular NLP packages to get the POS tags of a given sentence.

## Lecture Content
* Lec1: Part-of-Speach Tags
* Lec2: POS Tags Using spaCy
* Lec3: POS Tags with Hidden Markov Models
* Lec4: More POS tagging Ideas

## Summary
* These notes for Week 06 content covers POS tagging as a sequence labeling problem, moving from tag definitions to practical tools (spaCy), classic probabilistic models (HMM), and modern alternatives (CRF, unsupervised methods).

**Most testable contrasts:**
- Syntactic vs semantic roles of POS tags
- spaCy attributes: .pos_ vs .tag_ vs .dep_
- HMM assumptions: hidden (tags) vs observed (words); Markov dependence on tags, not words
- State space scaling: exponential in n-gram order
- Performance benchmarks: 90.3% → 93.7% → 96.7% → 97+%
- Domain shift: up to 3x error rate

---

## 1. PART-OF-SPEECH TAGS: DEFINITIONS AND CLASSES

### Core Definition
- Parts-of-speech are **syntactic classes** of words, not semantic meanings.
- They describe grammatical role (e.g., subject, object, modifier) rather than what the word means.

### Open (Lexical) vs Closed (Functional) Classes

| Class | Characteristics | Examples |
|-------|----------------|----------|
| Open (lexical) | New words easily added; content-bearing | Nouns, verbs, adjectives, adverbs, numbers |
| Closed (functional) | Fixed, limited set; grammatical glue | Determiners, conjunctions, pronouns, auxiliary verbs, prepositions, particles |

**Quiz-ready statement:** Open-class words admit new members (e.g., "to google" as a verb), while closed-class words rarely change.

### Common Penn Treebank Tags (Fine-Grained)

| Tag | Description | Example |
|-----|-------------|---------|
| NN | common singular noun | cat, investment |
| NNS | common plural noun | cats, rates |
| NNP | proper singular noun | IBM, Italy |
| NNPS | proper plural noun | Americans |
| JJ | adjective | fast, ill-mannered |
| JJR | comparative adjective | faster |
| JJS | superlative adjective | fastest |
| RB | adverb | quickly, occasionally |
| RBR | comparative adverb | faster (as adv) |
| RBS | superlative adverb | fastest (as adv) |
| VB | base verb | see, take |
| VBD | past tense verb | saw, took |
| VBG | present participle / gerund | seeing, taking |
| VBN | past participle | seen, taken |
| VBP | present verb, non-3rd person | see, take (I/you/we) |
| VBZ | present verb, 3rd person singular | sees, takes |
| DT | determiner | the, a, an |
| IN | preposition / subord. conjunction | in, on, of, that (as conj) |
| PRP | personal pronoun | I, you, he, we |
| PRP$ | possessive pronoun | my, your, his |
| CD | cardinal number | 1, 100, one |
| MD | modal auxiliary | can, may, will |
| CC | coordinating conjunction | and, or, but |
| TO | "to" as infinitive or preposition | to go, to the store |
| UH | interjection | oh, wow |

### High-Risk Confusable Pairs (Quiz Traps)

| Confusing Pair | Key Distinction |
|----------------|------------------|
| VBD vs VBN | VBD = past tense (I saw); VBN = past participle (I have seen) |
| NNP vs NN | NNP = proper noun (IBM); NN = common noun (company) |
| VBZ vs NNS | Both often end in -s; VBZ is verb (raises), NNS is noun (rates) |
| JJ vs RB | Some words (fast, hard) can be either; check what they modify |

---

## 2. AMBIGUITY AND CONTEXT

The same word can have multiple POS tags depending on context. POS tagging is **not** a dictionary lookup problem.

**Example from lecture:** "Fed raises interest rates 0.5 percent"
- Fed: NNP (proper noun) or VBD (past tense of feed) – context resolves to NNP.
- raises: VBZ (verb) or NNS (plural noun) – context resolves to VBZ.

**Factors that disambiguate (explicit in lecture):**
- Grammar (syntactic rules)
- Suffixes (e.g., -ing often VBG)
- Capitalization (e.g., "May" as NNP vs month)
- Gazetteers (name databases)

**Likely multiple-choice trap:** A question may say "POS tagging can be done by looking up each word in a dictionary." This is FALSE because of ambiguity.

---

## 3. WHY POS TAGGING? APPLICATIONS (FREQUENTLY TESTED)

POS tagging is a keystone preprocessing step for many NLP tasks. The lecture lists these applications:

| Application | Why POS is needed |
|-------------|--------------------|
| Text-to-speech | Pronunciation depends on tag (e.g., "record" as noun /ˈrɛkərd/ vs verb /rɪˈkɔrd/) |
| Lemmatization | Root form changes with tag (e.g., "saw" → see (VB) vs saw (NN)) |
| Noun-phrase chunking | Identify spans like `{JJ | NN}* {NN | NNS}` |
| Regex pattern extraction | Find patterns in tagged text (e.g., adjective-noun sequences) |
| Parsing | POS tags are input to syntactic parsers |

**Test-ready statement:** POS tagging is useful both as an end task and as a preprocessing step for higher-level syntactic analysis.

---

## 4. SPACY: PRACTICAL POS TAGGING AND NOUN CHUNKS

### Key Attributes (Highly Testable)

| Attribute | Description | Example Output |
|-----------|-------------|----------------|
| `token.pos_` | Coarse-grained POS tag | NOUN, VERB, ADJ, PROPN |
| `token.tag_` | Fine-grained Penn Treebank tag | NNP, VBD, JJ |
| `token.dep_` | Syntactic dependency relation | nsubj, dobj, amod |

**Code example from lecture:**
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
print([token.tag_ for token in doc])
# ['JJ', 'NNS', 'VBP', 'NN', 'NN', 'IN', 'NNS']
```

**Quiz trap:** Do not confuse .tag_ with .dep_. .tag_ gives POS; .dep_ gives dependency.

### Noun Phrase Chunking
- Access via `doc.noun_chunks`
- Groups tokens that form a base noun phrase (not the same as POS tagging)
- Example output: "Autonomous cars", "insurance liability", "manufacturers"

**Test distinction:** Noun chunking uses POS tags as input but is a different task. A quiz may ask: "Which is NOT a POS tagging task?" with noun chunking as a distractor.

---

## 5. HIDDEN MARKOV MODEL (HMM) FOR POS TAGGING

### Model Structure
- **Hidden sequence s** = POS tags (unobserved)
- **Observed sequence w** = words
- Joint probability for a sentence of length n:

$$P(s, w) = \prod_{i=1}^{n} P(s_i | s_{i-1}) P(w_i | s_i)$$

For a **trigram HMM** (second order), the state depends on two previous tags:

$$P(s, w) = \prod_{i=1}^{n} P(s_i | s_{i-1}, s_{i-2}) P(w_i | s_i)$$

Boundary conditions: START states provide $s_{-1}$ and $s_0$ for the first positions.

### HMM Assumptions (Explicit from Lecture)
1. The POS tag sequence is generated by a Markov model (limited history).
2. Hidden states are POS tag n-grams (bigram = tags; trigram = tag pairs).
3. Words are chosen independently, **conditioned only on the current tag/state**.

**Critical nuance:** The Markov assumption applies to **tags**, not to words. Words are conditionally independent given their tags.

### States and Order

| Tagger Order | State Representation | Number of States (|T| tags) |
|--------------|----------------------|------------------------------|
| Bigram (1st) | Current tag $t_i$ | $|T|$ |
| Trigram (2nd) | Tag pair $(t_{i-1}, t_i)$ | $|T|^2$ |
| 4-gram (3rd) | Triple $(t_{i-2}, t_{i-1}, t_i)$ | $|T|^3$ |

**Exponential scaling:** For order k, the number of possible state histories grows as $|T|^k$. This is exponential in k and causes sparsity.

### Transition Probabilities
- $P(t_i | t_{i-1}, t_{i-2})$ for trigram model
- Estimated from counts in labeled training data:
  $$\hat{P}(t_i | t_{i-1}, t_{i-2}) = \frac{\text{count}(t_{i-2}, t_{i-1}, t_i)}{\text{count}(t_{i-2}, t_{i-1})}$$

### Smoothing for Transitions (Linear Interpolation)
Because many tag triples are rare or unseen:
$$P(t_i | t_{i-1}, t_{i-2}) = \lambda_2 \hat{P}(t_i | t_{i-1}, t_{i-2}) + \lambda_1 \hat{P}(t_i | t_{i-1}) + (1 - \lambda_1 - \lambda_2) \hat{P}(t_i)$$

Weights $\lambda$ are tuned on held-out data (not test set, not training set).

### Emission Probabilities
- $P(w_i | t_i)$ = probability of word given tag
- Estimated from word-tag counts: $\hat{P}(w | t) = \frac{\text{count}(t, w)}{\text{count}(t)}$

### Unknown Word Problem
- Tags are closed-world (small fixed set)
- **Words are open-world** – new words appear constantly
- Unknown words affect **emission probabilities**, not transitions

**Solutions from lecture:**
- Replace rare/unseen words with `<unk>` token during training
- Use unknown-word classes based on:
  - Prefixes (e.g., first 3 characters)
  - Suffixes (e.g., last 3 characters)
  - Capitalization pattern (e.g., first letter capital)
  - Word shape (e.g., "Xx" for "Cat", "xxx" for "cat")

**Quiz trap:** A question may say "Unknown words are a problem for transition probabilities." This is FALSE; they affect emissions.

### Decoding (Finding the Best Tag Sequence)
- Represent all possible tag sequences as a **state lattice (trellis)** with START and END states.
- START and END are artificial states that handle sentence boundaries; they emit no words.
- Decoding = finding the highest-probability path through the lattice (using Viterbi algorithm).

**Important distinction:**
- **Training**: Estimate transition and emission probabilities from labeled data (with smoothing).
- **Decoding**: Apply pre-estimated parameters to find tag sequence for new, unlabeled sentence.

### Performance Benchmarks (WSJ Dataset)

| Model | Accuracy |
|-------|----------|
| Most common tag (bad unknown model) | 90.3% |
| Most common tag (good unknown word model) | 93.7% |
| TnT (carefully smoothed trigram tagger) | 96.7% |
| State of the art (modern neural/CRF) | 97+% |

**Annotation ceiling:** The benchmark Wall Street Journal dataset has about **2% human annotation error**. Therefore, 98% is practically perfect.

**Key insight:** The jump from 90.3% to 93.7% comes entirely from better unknown-word handling, not from modeling tag context.

---

## 6. BEYOND HMM: CONTEXT-FREE CLASSIFIERS AND CRF

### Context-Free Classifier for $P(tag | word)$
- Features (from lecture): word itself, lowercased word, prefixes, suffixes, capitalization, word shape.
- Achieves **93.7% accuracy without any context** (same as baseline with good unknown model).
- This shows that unknown-word features are more important than tag context for this baseline.

### Conditional Random Field (CRF) – Discriminative Sequence Model

**High-level description from lecture:**
- Extends hidden states from discrete tags to **continuous vector representations**.
- Builds a classifier to predict the tag based on the vector representation.
- Decodes the tag sequence by considering the joint transition-emission probability distribution.

**Key contrast with HMM:**

| Aspect | HMM | CRF |
|--------|-----|-----|
| Type | Generative ($P(s,w)$) | Discriminative ($P(s|w)$) |
| Independence | Words independent given tags | No independence assumption; can use rich overlapping features |
| Training | Maximum likelihood of joint | Maximize conditional likelihood |
| Context | Only from previous tags | Can use arbitrary features from whole sequence |

**Quiz-ready statement:** CRF is a discriminative alternative to HMM that can incorporate richer features and does not assume conditional independence of words.

---

## 7. UNSUPERVISED POS TAGGING AND DOMAIN SHIFT

### Unsupervised Approaches (No Labeled Data)
- Start with a uniform HMM and run **Expectation-Maximization (EM)**.
- Clustering based on distributional contexts (similar words appear in similar tag contexts).
- Use contextualized representations from pretrained neural language models.

### Domain Shift (Highly Testable)

- Tagger trained on one domain (e.g., Wall Street Journal news) applied to a different domain (e.g., Twitter).
- **Error rate can increase up to 3x**.
- Mistakes often concentrate on the domain-specific words you care about most (e.g., protein names in biomedical text).
- **Solution:** Train domain-specific taggers or fine-tune on in-domain data.

**Likely quiz question:** What happens when a news-trained POS tagger is used on Twitter data?  
**Answer:** Error rate can increase up to 3 times, especially on domain-relevant terms.

---

## 8. COMMON QUIZ TRAPS (VERDICT TABLE)

| Statement | Verdict | Correct Logic |
|-----------|---------|----------------|
| "POS tags represent the meaning of a word." | FALSE | POS tags are syntactic classes, not semantic. |
| "spaCy's .dep_ attribute gives fine-grained POS tags." | FALSE | .dep_ gives dependency relations; .tag_ gives fine-grained POS. |
| "In an HMM, words are conditionally independent given the previous word." | FALSE | Words are conditionally independent given the **tag**, not the previous word. |
| "The START state has a corresponding observed word." | FALSE | START is artificial; no word is emitted. |
| "Unseen words affect transition probabilities." | FALSE | Unseen words affect emission probabilities. |
| "Higher-order HMMs always outperform lower-order ones." | FALSE | Higher order suffers more sparsity; smoothing required to realize gains. |
| "Smoothing is only needed for emissions." | FALSE | Smoothing is needed for transitions (rare tag sequences) and emissions (rare word-tag pairs). |
| "Noun phrase chunking is the same as POS tagging." | FALSE | Chunking uses POS tags as input but is a separate grouping task. |
| "A CRF is generative like an HMM." | FALSE | CRF is discriminative; HMM is generative. |
| "The number of states in a trigram HMM grows quadratically with tag set size." | FALSE | It grows quadratically for bigram, but for trigram it is cubic ($|T|^3$). Exponential in order. |

---

## 9. QUANTITATIVE PRACTICE PROBLEMS

**Q1: Parameter count in bigram HMM**  
Suppose there are 45 POS tags (including START). How many transition probabilities must be estimated?  
**A:** From START: 45. From each tag to each tag: 45 * 45 = 2025. Total = 2070.

**Q2: State space growth**  
If tag set size is 45, how many possible bigram state histories? Trigram?  
**A:** Bigram (states = tags): 45. Trigram (states = tag pairs): 45^2 = 2025. 4-gram (states = tag triples): 45^3 = 91,125. This demonstrates exponential growth.

**Q3: Smoothing interpolation example**  
Suppose $P(VBZ | NN, DT)$ is unseen (count = 0). $\lambda_2 = 0.6$, $\lambda_1 = 0.3$, $(1-\lambda_1-\lambda_2)=0.1$. Bigram $P(VBZ|NN) = 0.2$, unigram $P(VBZ) = 0.05$. Compute smoothed probability.  
**A:** $0.6 \times 0 + 0.3 \times 0.2 + 0.1 \times 0.05 = 0.06 + 0.005 = 0.065$.

**Q4: Unknown word impact**  
If a test sentence contains the word "deepfake" which never appeared in training, which probability is affected?  
**A:** Emission probabilities $P(\text{deepfake} | t)$ for all tags $t$. Transitions remain unchanged.

---

## 10. ULTRA-CONDENSED EXAM SHEET (FINAL REVIEW)

- **POS = syntactic class**, not semantic.
- **Open class** = nouns, verbs, adjectives, adverbs, numbers (new words easily added).
- **Closed class** = determiners, conjunctions, pronouns, aux, preps, particles (fixed).
- **Ambiguity** = same word, different tags by context.
- **spaCy**: `.pos_` (coarse), `.tag_` (fine), `.dep_` (dependency). `doc.noun_chunks` for noun phrases.
- **HMM**: hidden = tags, observed = words. Joint $P(s,w) = \prod P(s_i|s_{i-1}) P(w_i|s_i)$.
- **Bigram state** = tag; **trigram state** = tag pair.
- **Transitions** from tag counts; **emissions** from word-tag counts.
- **Unknown words**: handle with `<unk>` or affix/shape features. Affect emissions, not transitions.
- **Decoding** = Viterbi path through lattice with START/END.
- **Performance**: 90.3% (bad unk) → 93.7% (good unk) → 96.7% (TnT) → 97+% (SOTA).
- **Annotation ceiling**: ~2% human error on WSJ, so 98% is near perfect.
- **Context-free features** (word shape, prefixes, etc.) reach 93.7% without context.
- **CRF**: discriminative, continuous representations, no independence assumptions.
- **Domain shift**: up to 3x error rate when moving domains (e.g., news → tweets).
