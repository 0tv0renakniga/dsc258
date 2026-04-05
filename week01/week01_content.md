# Tokenization, Stemming, and Lemmatization
* In this week, we will introduce a few popular NLP packages to preprocess the data, so it will be better understandable to machines. We will talk about examples in English and Chinese, so the students will have comprehensive ideas about the importance of preprocessing.

* **Goal:** By the end of this week you will be able to pre-process the text data in computer.

---

# What is NLP?

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [Outline](#outline)
- [Mining Knowledge from Big Data](#mining-knowledge-from-big-data)
- [Massive Unstructured Text Data](#massive-unstructured-text-data)
- [Goal: Texts, Knowledge & Insights](#goal-texts-knowledge--insights)
- [What Is Natural Language Processing](#what-is-natural-language-processing)
- [Example NLP Tasks](#example-nlp-tasks)
- [Sentiment Analysis as an Example](#sentiment-analysis-as-an-example)
- [Glossary](#glossary)

## Course and Lecture Context

### Lecture Title
- **What Is NLP?**

## Outline

### Course Information
- The lecture begins with **course information**.

### Main Topics
- **What is Natural Language Processing (NLP)?**
- **Pre-processing in NLP**
  - **Tokenization**
  - **Stemming & Lemmatization**
  - **The key: how to define “terms”?**

### Structural Note
- The outline indicates that the lecture is not limited to defining NLP; it also foreshadows important early-stage text processing issues, especially how text is broken into meaningful units and normalized for analysis.

## Mining Knowledge from Big Data

### Core Idea
- A central motivation of NLP is to help transform **Big Data** into **Structured Knowledge & Insights**.

### Process Flow
- The lecture presents the following transformation:

$$
\text{Big Data} \rightarrow \text{Structured Knowledge \& Insights}
$$

### Interpretation
- The emphasis is that raw data, by itself, is not the final goal.
- The desired outcome is **knowledge** and **insight** that can be organized, interpreted, and used.

## Mining Knowledge from Big Data

### Extended View of Data Sources
- The lecture expands the previous idea by distinguishing between:
  - **(Semi) Structured Data**
  - **Unstructured Text Data**

### Goal
- Both types of data contribute toward the same end goal:
  - **Structured Knowledge & Insights**

### Process Relationship
- The conceptual flow shown is:

$$
\text{(Semi) Structured Data} \; / \; \text{Unstructured Text Data} \rightarrow \text{Structured Knowledge \& Insights}
$$

### Diagram Description
- One slide uses an *iceberg-style diagram*:
  - The visible top portion is labeled **(Semi) Structured Data**
  - The much larger submerged portion is labeled **Unstructured Text Data**
  - An arrow points from the iceberg toward a representation of **knowledge/insight**
- This visual implies that **unstructured text data forms a large and important share of available information**.

## Massive Unstructured Text Data

### Examples of Unstructured Text Sources
- **News**
- **Social Media**
- **Business & Finance**
- **Scientific Papers**
- **Medical Records**
- **……**

### Meaning
- The lecture emphasizes that text data is **massive**, widespread, and drawn from many domains.
- These examples show that NLP is broadly applicable across public communication, enterprise data, scientific knowledge, and healthcare documentation.

### Figure Description
- The slide visually groups representative examples of:
  - News organizations
  - Social media platforms
  - Business and finance materials
  - Scientific content
  - Medical record content
- The figure reinforces that NLP must handle text from **many different sources and formats**.

## Goal: Texts, Knowledge & Insights

### Workflow
- The lecture presents a text-focused processing pipeline:

$$
\text{Unstructured Text Data} \rightarrow \text{Automated Models} \rightarrow \text{Structured Knowledge \& Insights}
$$

### Components
- **Unstructured Text Data**
- **Automated Models**
- **Structured Knowledge & Insights**

### Interpretation
- NLP systems rely on **automated models** to convert raw text into outputs that are more structured and useful.
- The focus is on making free-form human language computationally actionable.

### Diagram Description
- The visual depicts:
  - Unstructured text as scattered text elements
  - A model-driven transformation step
  - A final organized representation associated with **insight**
- This figure illustrates how computational processing creates order from textual complexity.

## What Is Natural Language Processing

### Definition
- **Natural Language Processing (NLP)** is defined in the lecture using a Wikipedia-based description:

> “Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence […], in particular how to program computers to process and analyze large amounts of natural language data.”

### Related Terms
- NLP is described as similar to:
  - **Text mining**
  - **Text analytics**
  - *etc.*

### Explanation
- This definition highlights that NLP is interdisciplinary.
- It combines ideas from language, computation, and AI to enable machines to work with human language at scale.

## Example NLP Tasks

### Common Tasks Mentioned
- **Classify the review’s sentiment**
- **Categorize a book based on its description**
- **Extract phrases from texts**
- **Recognize entities from texts**
- **Retrieve relevant documents based on query texts**
- **…**

### Task Interpretation
- These examples span several important classes of NLP problems:
  - **Classification**, such as sentiment labeling
  - **Categorization**, such as assigning content to topics or genres
  - **Information extraction**, such as phrase extraction
  - **Entity recognition**, such as finding names of people, places, organizations, or related entities
  - **Information retrieval**, such as finding documents relevant to a text query

### Importance
- The lecture uses this list to show that NLP is not a single task.
- Instead, it is a family of methods for understanding, organizing, and retrieving information from language.

## Sentiment Analysis as an Example

### Core Question
- The lecture poses the following question:

#### Given a review document, can we estimate its sentiment from its content?

### Definition
- **Sentiment analysis** is the task of inferring the emotional or evaluative tone of a text, such as whether a review is positive, negative, or mixed.

### Example Framing
- The slide uses **review documents** as the motivating example.
- The task is to use the **content of the review itself** to estimate sentiment.

### Figure Description
- The figure shows two product review examples:
  - One review is clearly **positive**, with a high star rating and favorable title text
  - Another review is **mixed or negative**, with a lower rating and critical wording
- This visual demonstrates that textual clues in reviews can be used to predict sentiment.

### Why This Example Matters
- Sentiment analysis is a useful introductory NLP task because:
  - It is intuitive
  - It uses natural text written by people
  - It clearly connects text content to a structured output label

## Glossary

### **Automated Models**
- Computational methods or systems used to transform **unstructured text data** into **structured knowledge and insights**.

### **Big Data**
- Large-scale data resources that can be transformed into **structured knowledge and insights**.

### **Entity Recognition**
- An NLP task involving the identification of entities from text.

### **Knowledge & Insights**
- The structured end products that result from processing raw or unstructured data.

### **Lemmatization**
- A pre-processing topic listed in the lecture outline, grouped with stemming as a method for normalizing text.

### **Natural Language Processing (NLP)**
- A subfield of linguistics, computer science, and artificial intelligence concerned with programming computers to process and analyze large amounts of natural language data.

### **Pre-processing in NLP**
- Early-stage NLP processing steps introduced in the outline, including **tokenization**, **stemming**, **lemmatization**, and decisions about how to define **terms**.

### **Sentiment Analysis**
- The task of estimating the sentiment of a review document from its textual content.

### **Stemming**
- A pre-processing topic listed in the lecture outline, grouped with lemmatization as a method for reducing or normalizing word forms.

### **Structured Knowledge & Insights**
- The organized information produced from raw data or unstructured text through computational processing.

### **Term**
- A meaningful unit used in text processing, highlighted in the outline through the question of how to define “terms.”

### **Text Analytics**
- A related term mentioned as similar to NLP.

### **Text Mining**
- A related term mentioned as similar to NLP.

### **Tokenization**
- A pre-processing step in NLP concerned with breaking text into units or tokens.

### **Unstructured Text Data**
- Free-form text data, such as news, social media, business and finance text, scientific papers, and medical records, which can be processed into structured knowledge and insights.

---

# NLP Pipelines

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [NLP Pipelines](#nlp-pipelines)
- [A Typical Text Mining Pipeline](#a-typical-text-mining-pipeline)
- [Pipeline Stages](#pipeline-stages)
- [Glossary](#glossary)

## Course and Lecture Context

### Lecture Title
- **NLP Pipelines**

## NLP Pipelines

### Main Topic
- The lecture section introduces **NLP pipelines**.

### Purpose
- The focus is on the structure of a **typical text mining pipeline**.
- The pipeline is presented as a staged workflow for transforming text into a form suitable for downstream modeling.

## A Typical Text Mining Pipeline

### Pipeline Overview
- The lecture presents the following sequence:

$$
\text{Preprocessing} \rightarrow \text{Representation} \rightarrow \text{Model Training}
$$

### High-Level Meaning
- The pipeline is organized into three major stages:
  - **Preprocessing**
  - **Representation**
  - **Model Training**

### Structural Interpretation
- The sequence implies that text mining is not done in a single step.
- Instead, raw text is progressively refined, encoded, and used for a task-specific model.

## Pipeline Stages

### Preprocessing
- **Preprocessing** is described as:

> Try to make texts “better formatted”

#### Interpretation
- This stage focuses on improving the form of the input text before more formal computational handling.
- The phrase “better formatted” suggests cleaning, standardizing, or otherwise preparing text so later stages can use it more effectively.

### Representation
- **Representation** is described as:

> Make the text data machine-actionable

#### Interpretation
- This stage converts text into a representation that a machine can work with directly.
- The core idea is that raw language must be transformed into a computational form before model learning can occur.

### Model Training
- **Model Training** is described as:

> Specifically for the end-task

#### Interpretation
- This stage is task-oriented.
- The model is trained with the specific downstream objective in mind, meaning the final modeling step depends on what problem is being solved.

## Glossary

### **Model Training**
- The pipeline stage in which a model is trained **specifically for the end-task**.

### **NLP Pipelines**
- A staged workflow in natural language processing for moving from text input to task-specific modeling.

### **Preprocessing**
- The pipeline stage that tries to make texts **“better formatted.”**

### **Representation**
- The pipeline stage that makes text data **machine-actionable**.

### **Text Mining Pipeline**
- A typical workflow consisting of **Preprocessing**, **Representation**, and **Model Training**.

---

# NLP Preprocessing

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [Outline](#outline)
- [Tokenization](#tokenization)
- [Stemming and Lemmatization](#stemming-and-lemmatization)
- [Popular Python Libraries](#popular-python-libraries)
- [NLTK Functionality](#nltk-functionality)
- [NLTK Tokenization Examples](#nltk-tokenization-examples)
- [NLTK Stemming](#nltk-stemming)
- [NLTK End-to-End Preprocessing Function](#nltk-end-to-end-preprocessing-function)
- [spaCy Functionality](#spacy-functionality)
- [spaCy Tokenization and Lemmatization Examples](#spacy-tokenization-and-lemmatization-examples)
- [spaCy End-to-End Preprocessing Function](#spacy-end-to-end-preprocessing-function)
- [Stemming vs. Lemmatization](#stemming-vs-lemmatization)
- [Glossary](#glossary)

## Course and Lecture Context

### Lecture Title
- **Pre-processing**

## Outline

### Topics Listed
- What’s Natural Language Processing (NLP)?
- **Pre-processing in NLP**
  - **Tokenization**
  - **Stemming & Lemmatization**
  - **The key: how to define “terms”?**

### Interpretation
- This lecture focuses specifically on **pre-processing in NLP**.
- The outline emphasizes that a central issue in preprocessing is deciding how to define textual **terms**.

## Tokenization

### Main Challenges
- **Language Challenges:** There are no spaces in some languages, for example **Chinese**.

### English-Specific Notes
- For English, tokenization is more complicated than `string.split()`.
- It involves:
  - Text into **sentences** tokenization.
  - Sentences into **words** tokenization.

### Instructional Note
- *Will see some examples soon!*

### Key Idea
- Tokenization is not merely splitting on whitespace.
- It requires handling sentence boundaries, punctuation, contractions, and language-specific structure.

## Stemming and Lemmatization

### Definition
- **Stemming & Lemmatization** reduce inflectional forms and sometimes derivationally related forms of a word to a common base form.

### Properties
- Mostly **rule-based**.
- Example algorithm:
  - **Porter’s algorithm (Porter, 1980)**

### Examples
| Original Form(s) | Base Form |
|---|---|
| am, is, are | be |
| car, cars, car’s, cars’ | car |
| president, presidential | president |
| food, foods | food |
| the boy's cars are different colors | the boy car be differ color |

### Interpretation
- The examples show that normalization can apply to both individual words and entire sentences.
- The final sentence example illustrates that normalized output may be less grammatical but more uniform for computation.

## Popular Python Libraries

### Libraries Mentioned
- **NLTK**
- **spaCy**
- **...**

### Example Setup
```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
```

### Purpose
- These libraries are presented as commonly used tools for preprocessing tasks such as tokenization, stemming, and lemmatization.

## NLTK Functionality

### Capabilities
- **Word tokenization:** More than `string.split()`.
- **Sentence tokenization/segmentation:** More than punctuation.
- **Stemming:** **Porter Stemmer** (probably the most classical stemmer).

### Combined Pipeline
- **NLTK Putting Things Together!**
  - Sentence Tokenization $\rightarrow$ Word Tokenization $\rightarrow$ Stemming

### Interpretation
- NLTK is shown as supporting the full preprocessing sequence from raw text to normalized tokens.

## NLTK Tokenization Examples

### Word Tokenization Example 1
```python
from nltk.tokenize import word_tokenize
text = "God is Great! I won a lottery."
print(word_tokenize(text))

# Output: ['God', 'is', 'Great', '!', 'I', 'won', 'a', 'lottery', '.']
```

### Output
```python
['God', 'is', 'Great', '!', 'I', 'won', 'a', 'lottery', '.']
```

### Word Tokenization Example 2
```python
text = "You're the best! Boy's running. The boy's car is awesome! :P :(("
print(word_tokenize(text))
```

### Output
```python
['You', "'re", 'the', 'best', '!', 'Boy', "'s", 'running', '.', 'The', 'boy', "'s", 'car', 'is', 'awesome', '!', ':', 'P', ':', '(', '(']
```

### What These Examples Show
- Tokenization separates punctuation into standalone tokens.
- Contractions and possessives are split into component parts such as `"You"` and `"'re"`, or `"boy"` and `"'s"`.
- Emoticon-like text is also split into tokens.

### Sentence Tokenization Example 1
```python
from nltk.tokenize import sent_tokenize
text = "God is Great! I won a lottery."
print(sent_tokenize(text))

# Output: ['God is Great!', 'I won a lottery ']
```

### Actual Output Shown
```python
['God is Great!', 'I won a lottery.']
```

### Sentence Tokenization Example 2
```python
text = "The nuggests are priced at $4.99. Are you crazy??! I'm not sure what's the effect the dots....."
print(sent_tokenize(text))
```

### Output
```python
['The nuggests are priced at $4.99.', 'Are you crazy??!', "I'm not sure what's the effect the dots....."]
```

### What These Examples Show
- Sentence segmentation is more sophisticated than splitting on periods.
- The tokenizer preserves:
  - Monetary amounts like `$4.99`
  - Multiple punctuation marks like `??!`
  - Ellipsis-like sequences such as `.....`

## NLTK Stemming

### Porter Stemmer
- **Porter Stemmer** is described as probably the most classical stemmer.

### Example Code
```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = ["program", "programs", "programer", "programing", "programers"]

for w in words:
    print(w, " : ", ps.stem(w))
```

### Output
```python
program   : program
programs  : program
programer : program
programing: program
programers: program
```

### Interpretation
- Multiple spelling and inflection variants are mapped to the same stem: `program`.

## NLTK End-to-End Preprocessing Function

### Download and Imports
```python
import nltk
nltk.download('punkt') # downloads you a model

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()
```

### Function Definition
```python
# return a list of tokens
def pre_processing_by_nltk(doc, stemming = True, need_sent = False):
    # step 1: get sentences
    sentences = sent_tokenize(doc)
    # step 2: get tokens
    tokens = []
    for sent in sentences:
        words = word_tokenize(sent)
        # step 3 (optional): stemming
        if stemming:
            words = [ps.stem(word) for word in words]
        if need_sent:
            tokens.append(words)
        else:
            tokens += words
    return tokens
```

### Test Case
```python
test_case1 = "The nuggests are priced at $4.99. Business Are you fuzzy dity dizzy crazy??! I'm not sure what's the effe..."
print(pre_processing_by_nltk(test_case1, need_sent = True))
```

### Output
```python
[['the', 'nuggest', 'are', 'price', 'at', '$', '4.99', '.'],
 ['busi', 'are', 'you', 'fuzzi', 'diti', 'dizzi', 'craz i', '?', '?', '!'],
 ['i', "'m", 'not', 'sure', 'what', "'s", 'the', 'effect', 'the', 'dot', '.....']]
```

### Pipeline Behavior
- Step 1: Sentence tokenization.
- Step 2: Word tokenization.
- Step 3: Optional stemming.
- If `need_sent = True`, the output preserves sentence-level grouping.
- Otherwise, all tokens are flattened into one list.

### NLTK Putting Things Together
- Sentence Tokenization
- Word Tokenization
- Stemming

## spaCy Functionality

### Core Notes
- **Tokenization:** spaCy wraps everything into objects so it looks different.
- **Lemmatization:** spaCy wraps everything into objects so it looks different.

### Combined Pipeline
- **spaCy Putting Things Together!**
  - Sentence Tokenization $\rightarrow$ Word Tokenization $\rightarrow$ Lemmatization

### Interpretation
- Unlike NLTK’s more direct token outputs, spaCy emphasizes object-oriented document and token structures.

## spaCy Tokenization and Lemmatization Examples

### Load spaCy Model
```python
import spacy
nlp_processor = spacy.load("en_core_web_sm")
```

### Create a Document
```python
text = "God is Great! I won a lottery."
doc = nlp_processor(text)
print(type(doc))
```

### Output
```python
<class 'spacy.tokens.doc.Doc'>
```

### Sentence Iteration
```python
for sent in doc.sents:
    print(sent)
```

### Output
```python
God is Great!
I won a lottery.
```

### Token Iteration
```python
for word in doc:
    print(type(word))
    print(word.text)
```

### Output
```python
<class 'spacy.tokens.token.Token'>
God
<class 'spacy.tokens.token.Token'>
is
<class 'spacy.tokens.token.Token'>
Great
<class 'spacy.tokens.token.Token'>
!
<class 'spacy.tokens.token.Token'>
I
<class 'spacy.tokens.token.Token'>
won
<class 'spacy.tokens.token.Token'>
a
<class 'spacy.tokens.token.Token'>
lottery
<class 'spacy.tokens.token.Token'>
.
```

### Lemmatization Example
```python
for word in doc:
    print(word.text, word.lemma_)
```

### Output
```python
God God
is be
Great great
! !
I I
won win
a a
lottery lottery
. .
```

### What These Examples Show
- A spaCy document is a **Doc** object.
- Individual tokens are **Token** objects.
- Lemmatization can map inflected forms to dictionary-like base forms, such as:
  - `is` $\rightarrow$ `be`
  - `won` $\rightarrow$ `win`

## spaCy End-to-End Preprocessing Function

### Model Import and Load
```python
import en_core_web_sm
nlp_processor = en_core_web_sm.load()
```

### Function Definition
```python
# return a list of tokens
def pre_processing_by_spacy(doc, lemma = True, need_sent = False):
    doc_class = nlp_processor(doc)
    tokens = []
    # step 1: get sentences
    for sent in doc_class.sents:
        # step 2: get tokens
        # step 3 (optional): lemma
        if lemma:
            words = [token.lemma_ for token in sent]
        else:
            words = [token.text for token in sent]
        if need_sent:
            tokens.append(words)
        else:
            tokens += words
    return tokens
```

### Example Call
```python
pre_processing_by_spacy(test_case1, need_sent = True)
```

### Output
```python
[['the', 'nuggest', 'be', 'price', 'at', '$', '4.99', '.'],
 ['business', 'be', 'you', 'fuzzy', 'dity', 'dizzy', 'crazy', '?', '?', '!'],
 ['I', 'be', 'not', 'sure', 'what', 'be', 'the', 'effect', 'the', 'dot', '.....']]
```

### Pipeline Behavior
- Step 1: Sentence tokenization through spaCy’s document object.
- Step 2: Token extraction from each sentence.
- Step 3: Optional lemmatization via `token.lemma_`.
- Sentence grouping is optionally preserved using `need_sent`.

### spaCy Putting Things Together
- Sentence Tokenization
- Word Tokenization
- Lemmatization

## Stemming vs. Lemmatization

### Main Claim
- They are subtly different.

### Library Implementations
- **NLTK** does **stemming**.
- **spaCy** does **lemmatization**.

### Comparative Examples

#### Example 1
```python
pre_processing_by_nltk("good better best")
```

```python
['good', 'better', 'best']
```

```python
pre_processing_by_spacy("good better best")
```

```python
['good', 'well', 'good']
```

#### Interpretation
- spaCy lemmatization normalizes comparative and superlative forms toward lemma-like forms.
- NLTK stemming leaves these particular words unchanged in this example.

#### Example 2
```python
pre_processing_by_nltk("I finished the hw1. I will finish hw1.")
```

```python
['i', 'finish', 'the', 'hw1', '.', 'i', 'will', 'finish', 'hw1', '.']
```

```python
pre_processing_by_spacy("I finished the hw1. I will finish hw1.")
```

```python
['I', 'finish', 'the', 'hw1', '.', 'I', 'will', 'finish', 'hw1', '.']
```

#### Interpretation
- Both approaches normalize `finished` to `finish` in this example.

#### Example 3
```python
pre_processing_by_nltk("You were a programmer. I am a programmer.")
```

```python
['you', 'were', 'a', 'programm', '.', 'i', 'am', 'a', 'programm', '.']
```

```python
pre_processing_by_spacy("I was a programmer. I am a programmer.")
```

```python
['I', 'be', 'a', 'programmer', '.', 'I', 'be', 'a', 'programmer', '.']
```

> NOTE: The lecture examples are not perfectly parallel in subject wording across the two implementations. The NLTK example uses **“You were a programmer. I am a programmer.”** while the spaCy example uses **“I was a programmer. I am a programmer.”** Both are included here exactly as presented so the discrepancy can be reviewed.

#### Interpretation
- NLTK stemming reduces `programmer` to `programm`.
- spaCy lemmatization preserves `programmer` but maps verbal forms such as `was` and `am` to `be`.

### Overall Difference
- **Stemming** often applies heuristic or rule-based truncation and may produce stems that are not standard dictionary words.
- **Lemmatization** aims for linguistically grounded base forms and often produces more interpretable outputs.

## Glossary

### **Lemmatization**
- The process of reducing inflectional forms, and sometimes derivationally related forms, of a word to a common base form; in this lecture, spaCy is used for lemmatization.

### **NLTK**
- A popular Python library for NLP that provides word tokenization, sentence tokenization, and stemming.

### **Porter Stemmer**
- A classical stemming algorithm associated with **Porter’s algorithm (Porter, 1980)**.

### **Pre-processing**
- Early-stage NLP work focused on making text suitable for later computational processing, including tokenization, stemming, and lemmatization.

### **Sentence Tokenization**
- The process of splitting text into sentences rather than merely splitting on punctuation.

### **spaCy**
- A popular Python library for NLP that wraps text into objects such as **Doc** and **Token**, and supports tokenization and lemmatization.

### **Stemming**
- The process of reducing word forms to a common stem, often using mostly rule-based methods and possibly producing non-dictionary outputs.

### **Term**
- A meaningful textual unit whose definition is highlighted in the lecture as a key issue in preprocessing.

### **Token**
- A unit of text produced by tokenization, such as a word, punctuation mark, or symbol.

### **Tokenization**
- The process of splitting text into units, including both sentence-level tokenization and word-level tokenization.

### **Word Tokenization**
- The process of splitting sentences into words or word-like units, going beyond simple whitespace-based splitting.

---

# How to Define Terms

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [Outline](#outline)
- [The Key in NLP Pre-processing](#the-key-in-nlp-pre-processing)
- [Zipf’s Law](#zipfs-law)
- [Zipf’s Law: Two Extremes](#zipfs-law-two-extremes)
- [Words vs. Phrases](#words-vs-phrases)
- [A Simple Phrase Mining Method](#a-simple-phrase-mining-method)
- [Summary](#summary)
- [Glossary](#glossary)

## Course and Lecture Context

### Lecture Title
- **How to Define Terms**

## Outline

### Topics Listed
- What’s Natural Language Processing (NLP)?
- Pre-processing in NLP
  - Tokenization
  - Stemming & Lemmatization
  - **The key: how to define “terms”?**

### Lecture Focus
- This lecture section emphasizes that one of the most important issues in NLP preprocessing is deciding what should count as a **term**.

## The Key in NLP Pre-processing

### Central Question
- **How to define “term”?**

### Desired Property of Terms
- Terms are expected to be **orthogonal**.
  - Ideally **“non-overlapping” in semantics**.

### Possible Choices for Terms
- **Words**
- **Stemmed words**
- **Phrases**
- **“Latent concepts”**
- **…**

### Interpretation
- The lecture frames term definition as a modeling choice.
- Different definitions of a term may capture meaning with different levels of precision and ambiguity.

## Zipf’s Law

### Definition
- **Zipf’s Law** describes the relationship between **word frequency** and **rank**.

### Core Relationship
- **Rank * Frequency ≈ Constant**

### Formula
$$
F(w) = \frac{C}{r(w)^\alpha}
$$

### Parameters Given
- $\alpha \approx 1$
- $C \approx 0.1$

### Graph Interpretation
- **Word Freq. (Y-axis) vs. Word Rank (X-axis)**
- The figure shows a power-law style curve:
  - Very high-frequency words appear at low ranks
  - Frequencies rapidly decrease as rank increases
  - A long tail extends toward many low-frequency words

### Frequency Regions
- **High Frequency:** Biggest data structure (**Stop words**).
- **Medium Frequency:** Most useful words.
- **Low Frequency:** Is “too rare” a problem?

### Generalized Zipf’s Law
$$
F(w) = \frac{C}{[r(w) + B]^\alpha}
$$

### Note on Scope
- **Generalized Zipf’s law** is described as applicable in many domains.

## Zipf’s Law: Two Extremes

### 1. The High End: Stopwords
- The most frequent words in a language, for example:
  - `a`
  - `an`
  - `the`
  - `…`

### Key Notes on Stopwords
- There is **no single universal list**.
- We typically assume that stopwords are **not useful** in NLP.
- They can be **removed in pre-processing**.

### 2. The Low End: Rare Words
- We typically replace too rare words by a special token **`<UNK>`**.

### Forward Reference
- *Will have a closer look in Week 2.*

### Interpretation
- The lecture highlights that both extremes of the frequency distribution can be problematic:
  - Very common words may add little semantic value.
  - Very rare words may be too sparse to model effectively.

## Words vs. Phrases

### Case Study
- **Analyzing US news articles on April 9, 2017**

### Before Phrase Mining: Ambiguity

#### Which “United”?
- **United States?**
- **United Parcel Service?**

#### What’s “Dao”?
- **A person?**
- **A place?**

### Meaning of the Problem
- Individual words can be ambiguous when viewed in isolation.
- A single token may not provide enough semantic context for reliable interpretation.

### Figure Description
- The slide includes a word-cloud-style visualization of news terms such as:
  - `dao`
  - `united`
  - `airline`
  - `plane`
  - `incident`
  - `chicago`
  - `passengers`
  - and related terms
- This figure illustrates how isolated words can surface important topics while still leaving ambiguity unresolved.

### After Phrase Mining: Clarity
- **United Express Flight 3411 incident**
- **United Airline**
- **David Dao:** A person

### Source Note
- *Source: NBC News, April 12, 2017*

### Figure Description
- The lecture includes an NBC News example referencing **David Dao**, described in the context of a news headline about court papers and United Airlines preserving evidence.
- This supports the point that phrase-level terms provide clearer semantic interpretation than isolated words.

### Interpretation
- Phrase mining helps convert ambiguous individual words into more meaningful, semantically coherent units.
- This can improve downstream understanding and reduce confusion in NLP systems.

## A Simple Phrase Mining Method

### Main Idea
- **Using a greedy merge algorithm!**

### Procedure
1. **Repeat:**
   - Find the most coherent adjacent pairs of “words”.
   - **Merge** them if the coherence measure is **above a threshold**.
2. **Until:** The measure is below a threshold.

### Example Evolution
- **Initial:** `Markov` | `Blanket` | `Feature` | `Selection` | `for` | `Support` | `Vector` | `Machine`
- **Merging:** `Markov Blanket` | `Feature Selection` | `for` | `Support Vector` | `Machine`

### Figure Description
- The slide includes a diagram showing adjacent words being combined into phrases, including:
  - `Markov + Blanket -> Markov Blanket`
  - `Feature + Selection -> Feature Selection`
  - `Support + Vector -> Support Vector`
  - followed by continued phrase formation toward `Support Vector Machine`

### Interpretation
- The algorithm is **greedy** because it repeatedly merges the best local adjacent pair.
- Phrase construction depends on a **coherence measure** and a **threshold**, so phrase growth stops when the evidence for merging becomes too weak.

## Summary

### Topics Revisited
- What’s Natural Language Processing (NLP)?
- Pre-processing in NLP
  - Tokenization
  - Stemming & Lemmatization
  - The key: how to define “terms”?

### Role of This Summary
- The summary slide restates that the lecture’s preprocessing discussion culminates in the central issue of **term definition**.

## Glossary

### **Generalized Zipf’s Law**
- A broader form of Zipf’s Law given by $F(w) = \frac{C}{[r(w) + B]^\alpha}$ and described as applicable in many domains.

### **Greedy Merge Algorithm**
- A phrase mining method that repeatedly merges the most coherent adjacent word pairs while the coherence measure remains above a threshold.

### **Latent Concepts**
- One of the possible choices for defining terms in NLP, beyond words, stemmed words, and phrases.

### **Orthogonal Terms**
- Terms that are ideally **non-overlapping in semantics**.

### **Phrase Mining**
- The process of identifying and merging word sequences into meaningful phrases to improve semantic clarity.

### **Phrases**
- Multi-word expressions that may represent meaning more clearly than isolated words.

### **Rare Words**
- Very low-frequency words that are often replaced by the special token **`<UNK>`**.

### **Stopwords**
- Very high-frequency words in a language, such as `a`, `an`, and `the`, which are typically assumed not to be useful in NLP and may be removed during preprocessing.

### **Term**
- A basic semantic unit used in NLP preprocessing, which may be defined as a word, stemmed word, phrase, latent concept, or another representation.

### **Words**
- One possible unit used to define terms in NLP.

### **Zipf’s Law**
- A law describing the relationship between word frequency and word rank, often approximated by a power-law formula.

---

# Zipf's Law and the Internet

## Table of Contents
- [Course and Lecture Context](#course-and-lecture-context)
- [Outline](#outline)
- [The Key in NLP Pre-processing](#the-key-in-nlp-pre-processing)
- [Zipf’s Law](#zipfs-law)
- [Zipf’s Law: Two Extremes](#zipfs-law-two-extremes)
- [Words vs. Phrases](#words-vs-phrases)
- [A Simple Phrase Mining Method](#a-simple-phrase-mining-method)
- [Summary](#summary)
- [Glossary](#glossary)

## Course and Lecture Context

### Lecture Title
- **How to Define Terms**

## Outline

### Topics Listed
- What’s Natural Language Processing (NLP)?
- Pre-processing in NLP
  - Tokenization
  - Stemming & Lemmatization
  - **The key: how to define “terms”?**

### Lecture Focus
- This lecture section emphasizes that one of the most important issues in NLP preprocessing is deciding what should count as a **term**.

## The Key in NLP Pre-processing

### Central Question
- **How to define “term”?**

### Desired Property of Terms
- Terms are expected to be **orthogonal**.
  - Ideally **“non-overlapping” in semantics**.

### Possible Choices for Terms
- **Words**
- **Stemmed words**
- **Phrases**
- **“Latent concepts”**
- **…**

### Interpretation
- The lecture frames term definition as a modeling choice.
- Different definitions of a term may capture meaning with different levels of precision and ambiguity.

## Zipf’s Law

### Definition
- **Zipf’s Law** describes the relationship between **word frequency** and **rank**.

### Core Relationship
- **Rank * Frequency ≈ Constant**

### Formula
$$
F(w) = \frac{C}{r(w)^\alpha}
$$

### Parameters Given
- $\alpha \approx 1$
- $C \approx 0.1$

### Graph Interpretation
- **Word Freq. (Y-axis) vs. Word Rank (X-axis)**
- The figure shows a power-law style curve:
  - Very high-frequency words appear at low ranks
  - Frequencies rapidly decrease as rank increases
  - A long tail extends toward many low-frequency words

### Frequency Regions
- **High Frequency:** Biggest data structure (**Stop words**).
- **Medium Frequency:** Most useful words.
- **Low Frequency:** Is “too rare” a problem?

### Generalized Zipf’s Law
$$
F(w) = \frac{C}{[r(w) + B]^\alpha}
$$

### Note on Scope
- **Generalized Zipf’s law** is described as applicable in many domains.

## Zipf’s Law: Two Extremes

### 1. The High End: Stopwords
- The most frequent words in a language, for example:
  - `a`
  - `an`
  - `the`
  - `…`

### Key Notes on Stopwords
- There is **no single universal list**.
- We typically assume that stopwords are **not useful** in NLP.
- They can be **removed in pre-processing**.

### 2. The Low End: Rare Words
- We typically replace too rare words by a special token **`<UNK>`**.

### Forward Reference
- *Will have a closer look in Week 2.*

### Interpretation
- The lecture highlights that both extremes of the frequency distribution can be problematic:
  - Very common words may add little semantic value.
  - Very rare words may be too sparse to model effectively.

## Words vs. Phrases

### Case Study
- **Analyzing US news articles on April 9, 2017**

### Before Phrase Mining: Ambiguity

#### Which “United”?
- **United States?**
- **United Parcel Service?**

#### What’s “Dao”?
- **A person?**
- **A place?**

### Meaning of the Problem
- Individual words can be ambiguous when viewed in isolation.
- A single token may not provide enough semantic context for reliable interpretation.

### Figure Description
- The slide includes a word-cloud-style visualization of news terms such as:
  - `dao`
  - `united`
  - `airline`
  - `plane`
  - `incident`
  - `chicago`
  - `passengers`
  - and related terms
- This figure illustrates how isolated words can surface important topics while still leaving ambiguity unresolved.

### After Phrase Mining: Clarity
- **United Express Flight 3411 incident**
- **United Airline**
- **David Dao:** A person

### Source Note
- *Source: NBC News, April 12, 2017*

### Figure Description
- The lecture includes an NBC News example referencing **David Dao**, described in the context of a news headline about court papers and United Airlines preserving evidence.
- This supports the point that phrase-level terms provide clearer semantic interpretation than isolated words.

### Interpretation
- Phrase mining helps convert ambiguous individual words into more meaningful, semantically coherent units.
- This can improve downstream understanding and reduce confusion in NLP systems.

## A Simple Phrase Mining Method

### Main Idea
- **Using a greedy merge algorithm!**

### Procedure
1. **Repeat:**
   - Find the most coherent adjacent pairs of “words”.
   - **Merge** them if the coherence measure is **above a threshold**.
2. **Until:** The measure is below a threshold.

### Example Evolution
- **Initial:** `Markov` | `Blanket` | `Feature` | `Selection` | `for` | `Support` | `Vector` | `Machine`
- **Merging:** `Markov Blanket` | `Feature Selection` | `for` | `Support Vector` | `Machine`

### Figure Description
- The slide includes a diagram showing adjacent words being combined into phrases, including:
  - `Markov + Blanket -> Markov Blanket`
  - `Feature + Selection -> Feature Selection`
  - `Support + Vector -> Support Vector`
  - followed by continued phrase formation toward `Support Vector Machine`

### Interpretation
- The algorithm is **greedy** because it repeatedly merges the best local adjacent pair.
- Phrase construction depends on a **coherence measure** and a **threshold**, so phrase growth stops when the evidence for merging becomes too weak.

## Summary

### Topics Revisited
- What’s Natural Language Processing (NLP)?
- Pre-processing in NLP
  - Tokenization
  - Stemming & Lemmatization
  - The key: how to define “terms”?

### Role of This Summary
- The summary slide restates that the lecture’s preprocessing discussion culminates in the central issue of **term definition**.

## Glossary

### **Generalized Zipf’s Law**
- A broader form of Zipf’s Law given by $F(w) = \frac{C}{[r(w) + B]^\alpha}$ and described as applicable in many domains.

### **Greedy Merge Algorithm**
- A phrase mining method that repeatedly merges the most coherent adjacent word pairs while the coherence measure remains above a threshold.

### **Latent Concepts**
- One of the possible choices for defining terms in NLP, beyond words, stemmed words, and phrases.

### **Orthogonal Terms**
- Terms that are ideally **non-overlapping in semantics**.

### **Phrase Mining**
- The process of identifying and merging word sequences into meaningful phrases to improve semantic clarity.

### **Phrases**
- Multi-word expressions that may represent meaning more clearly than isolated words.

### **Rare Words**
- Very low-frequency words that are often replaced by the special token **`<UNK>`**.

### **Stopwords**
- Very high-frequency words in a language, such as `a`, `an`, and `the`, which are typically assumed not to be useful in NLP and may be removed during preprocessing.

### **Term**
- A basic semantic unit used in NLP preprocessing, which may be defined as a word, stemmed word, phrase, latent concept, or another representation.

### **Words**
- One possible unit used to define terms in NLP.

### **Zipf’s Law**
- A law describing the relationship between word frequency and word rank, often approximated by a power-law formula.
Prepared using GPT-5.4
i've attached 05_zipfs_law_and_the_internet.pdf and here is the text:
[# Zipf’s law and the Internet
## Abstract
* Zipf's law governs many features of the Internet. Observations of Zipf distributions, while
interesting in and of themselves, have strong implications for the design and function of the Internet.
The connectivity of Internet routers influences the robustness of the network while the distribution in
the number of email contacts affects the spread of email viruses. Even web caching strategies are
formulated to account for a Zipf distribution in the number of requests for webpages.
* **Keywords: Zipf´s law, caching, networks**
## Introduction

The wide adoption of the Internet has fundamentally altered the ways in which we
communicate, gather information, conduct businesses and make purchases. As the use of the
World Wide Web and email skyrocketed, computer scientists and physicists rushed to
characterize this new phenomenon. While initially they were surprised by the tremendous
variety the Internet demonstrated in the size of its features, they soon discovered a widespread
pattern in their measurements: there are many small elements contained within the Web, but
few large ones. A few sites consist of millions of pages, but millions of sites only contain a
handful of pages. Few sites contain millions of links, but many sites have one or two. Millions
of users flock to a few select sites, giving little attention to millions of others.
This pattern has of course long been familiar to those studying distributions in income
(Pareto 1896), word frequencies in text (Zipf 1932), and city sizes (Zipf 1949). It can be
expressed in mathematical fashion as a power law, meaning that the probability of attaining a
certain size x is proportional to x -τ, where τ is greater than or equal to 1. Unlike the more
familiar Gaussian distribution, a power law distribution has no ‘typical’ scale and is hence
frequently called ‘scale-free’. A power law also gives a finite probability to very large
elements, whereas the exponential tail in a Gaussian distribution makes elements much larger
than the mean extremely unlikely. For example, city sizes, which are governed by a power
law distribution, include a few mega cities that are orders of magnitude larger than the mean
city size. On the other hand, a Gaussian, which describes for example the distribution of
heights in humans, does not allow for a person who is several times taller than the average.
Figure 1 shows a series of scale free distributions in the sizes of websites in terms of the
number of pages they include, the number of links given to or received from other sites and
the number of unique users visiting the site.
Below is the reformatted version of the provided excerpt from "Zipf’s Law and the Internet" by Lada A. Adamic and Bernardo A. Huberman. All technical data, mathematical formulas, and table values have been preserved as requested.
Zipf’s Law and the Internet: Statistical Distributions
Analysis of Web Measurements (1997)
I. Fitted Power Law Distributions
The following distributions characterize the scale-free nature of the early World Wide Web.
a) Number of Pages: Measured across crawls by infoseek.com and archive.org.
b) Number of Visitors: Binned distribution showing a power-law fit of $\tau = -2.07$.
c) Out-link Distribution: Number of links pointing from a site, with a $\tau = 2.0$ fit.
d) In-link Distribution: Number of links pointing to a site, with a $\tau = 2.0$ fit.
Figure 1 Note: Distributions measured in 1997 across various scales ($10^0$ to $10^6$).
II. Mathematical Recasting: PDF to Zipf Ranked Distribution
While data is often plotted as a Probability Density Function (PDF) to emphasize the count of small elements, it can be recast into a Zipf Ranked Distribution to highlight the largest elements.
The Conversion Logic:
PDF Formula: $p(x) = Cx^{-\tau}$
Probability of size $y$ or larger:
$$P(x > y) = \sum_{y}^{\infty} Cx^{-\tau} \approx Ay^{-\tau+1}$$
(Where $C$ and $A$ are constants)
Expected Rank ($r$): If there are $N$ total websites, the expected number of sites greater than $N$ is given by $r = NAy^{\tau-1}$.
Ranked Variable Size: Solving for $y$, the size of the $r^{th}$ ranked variable is proportional to $r^{-1/(\tau-1)} = r^{-\alpha}$, where $\alpha$ is the Zipf rank exponent.
III. User Traffic and Popularity
Figure 2: AOL Visitor Ranking
Sites ranked by the number of unique AOL visitors (Dec. 1, 1997). AOL was the largest ISP in the U.S. at the time. The fit follows a Zipf distribution: $n_r \sim r^{-1}$. The distribution shows mild concavity and a ranked exponent of 1.
Table 1: Distribution of User Volume
Determined by unique AOL visitors on Dec. 1, 1997, including general, adult, and .edu sites.
% of Sites
% User Volume
0.1%
32.36%
1%
55.63%
5%
74.81%
10%
82.26%
50%
94.92%
Key Observation: A small number of sites (e.g., Yahoo) are extremely popular and capture a disproportionate amount of total Internet traffic, while the bottom 50% of sites share roughly 5% of the total volume.
## ]

my boss and stakeholders are watching over my shoulder so make sure you do not omit anything and produce high quality markdown file

Answer skipped
Prepared using GPT-5.4
i've attached 05_zipfs_law_and_the_internet.pdf and here is the text:
[# Zipf’s law and the Internet
## Abstract
* Zipf's law governs many features of the Internet. Observations of Zipf distributions, while
interesting in and of themselves, have strong implications for the design and function of the Internet.
The connectivity of Internet routers influences the robustness of the network while the distribution in
the number of email contacts affects the spread of email viruses. Even web caching strategies are
formulated to account for a Zipf distribution in the number of requests for webpages.
* **Keywords: Zipf´s law, caching, networks**
## Introduction

The wide adoption of the Internet has fundamentally altered the ways in which we
communicate, gather information, conduct businesses and make purchases. As the use of the
World Wide Web and email skyrocketed, computer scientists and physicists rushed to
characterize this new phenomenon. While initially they were surprised by the tremendous
variety the Internet demonstrated in the size of its features, they soon discovered a widespread
pattern in their measurements: there are many small elements contained within the Web, but
few large ones. A few sites consist of millions of pages, but millions of sites only contain a
handful of pages. Few sites contain millions of links, but many sites have one or two. Millions
of users flock to a few select sites, giving little attention to millions of others.
This pattern has of course long been familiar to those studying distributions in income
(Pareto 1896), word frequencies in text (Zipf 1932), and city sizes (Zipf 1949). It can be
expressed in mathematical fashion as a power law, meaning that the probability of attaining a
certain size x is proportional to x -τ, where τ is greater than or equal to 1. Unlike the more
familiar Gaussian distribution, a power law distribution has no ‘typical’ scale and is hence
frequently called ‘scale-free’. A power law also gives a finite probability to very large
elements, whereas the exponential tail in a Gaussian distribution makes elements much larger
than the mean extremely unlikely. For example, city sizes, which are governed by a power
law distribution, include a few mega cities that are orders of magnitude larger than the mean
city size. On the other hand, a Gaussian, which describes for example the distribution of
heights in humans, does not allow for a person who is several times taller than the average.
Figure 1 shows a series of scale free distributions in the sizes of websites in terms of the
number of pages they include, the number of links given to or received from other sites and
the number of unique users visiting the site.
Below is the reformatted version of the provided excerpt from "Zipf’s Law and the Internet" by Lada A. Adamic and Bernardo A. Huberman. All technical data, mathematical formulas, and table values have been preserved as requested.
Zipf’s Law and the Internet: Statistical Distributions
Analysis of Web Measurements (1997)
I. Fitted Power Law Distributions
The following distributions characterize the scale-free nature of the early World Wide Web.
a) Number of Pages: Measured across crawls by infoseek.com and archive.org.
b) Number of Visitors: Binned distribution showing a power-law fit of $\\tau = -2.07$.
c) Out-link Distribution: Number of links pointing from a site, with a $\\tau = 2.0$ fit.
d) In-link Distribution: Number of links pointing to a site, with a $\\tau = 2.0$ fit.
Figure 1 Note: Distributions measured in 1997 across various scales ($10^0$ to $10^6$).
II. Mathematical Recasting: PDF to Zipf Ranked Distribution
While data is often plotted as a Probability Density Function (PDF) to emphasize the count of small elements, it can be recast into a Zipf Ranked Distribution to highlight the largest elements.
The Conversion Logic:
PDF Formula: $p(x) = Cx^{-\\tau}$
Probability of size $y$ or larger:
$$P(x > y) = \\sum_{y}^{\\infty} Cx^{-\\tau} \\approx Ay^{-\\tau+1}$$
(Where $C$ and $A$ are constants)
Expected Rank ($r$): If there are $N$ total websites, the expected number of sites greater than $N$ is given by $r = NAy^{\\tau-1}$.
Ranked Variable Size: Solving for $y$, the size of the $r^{th}$ ranked variable is proportional to $r^{-1/(\\tau-1)} = r^{-\\alpha}$, where $\\alpha$ is the Zipf rank exponent.
III. User Traffic and Popularity
Figure 2: AOL Visitor Ranking
Sites ranked by the number of unique AOL visitors (Dec. 1, 1997). AOL was the largest ISP in the U.S. at the time. The fit follows a Zipf distribution: $n_r \\sim r^{-1}$. The distribution shows mild concavity and a ranked exponent of 1.
Table 1: Distribution of User Volume
Determined by unique AOL visitors on Dec. 1, 1997, including general, adult, and .edu sites.
% of Sites
% User Volume
0.1%
32.36%
1%
55.63%
5%
74.81%
10%
82.26%
50%
94.92%
Key Observation: A small number of sites (e.g., Yahoo) are extremely popular and capture a disproportionate amount of total Internet traffic, while the bottom 50% of sites share roughly 5% of the total volume.
## A Growth Model
The pervasiveness of Zipf distributions on the Internet can be explained by an intuitive
growth model (Huberman 1999) that incorporates three simple assumptions. Let us formulate
the argu-ment in terms of the number of web pages hosted on a website. Similar arguments
can be applied just as easily to the number of visitors or links. The first assumption is that of
proportional growth or preferential attachment, i.e. the number of pages added to or removed
from the site is proportional to the number of pages already present. For example, a site with a
million pages might have a whole team of webmasters or generate its content automatically. It
could easily gain or shed a several thousand pages on any given day. On the other hand, it
would be surprising, but not impossible, for a site with only a handful of pages to suddenly
add a thousand more.
This multiplicative stochastic growth process yields a lognormal distribution in the
number of pages at a site after a fixed period of time. However, the World Wide Web is
anything but fixed. Its first decade was a period of rapid growth, with sites appearing at an
exponential rate. It so happens that when one computes an exponentially weighted mixture of
lognormals one obtains a power-law distribution exactly!
While the exponential growth in the number of websites and their stochastic addition of
pages alone can produce power law distributions, a key ingredient is still missing. For in spite
of the random nature of the growth, if one were taking a mixture of lognormals depending
only on a time variable, one would expect that the sites established early would have grown to
greater sizes than recently founded ones. However, studies have found only weak correlation
between the size of a site and its age (equivalently some very popular sites were founded
more recently, while sites present at the very start of the Internet boom did not necessarily
acquire a wide audience). The missing assumption is that sites can grow at different rates,
depending on the type of content and interest that they generate. Incorporating variability in
growth rates again yields power law distributions with varying exponents. The greater the
difference in growth rates among sites, the lower the exponent τ, which means that the
inequality in site sizes increases. In summary, a very simple assumption of stochastic
multiplicative growth, combined with the fact that sites appear at different times and/or grow
at different rates, leads to an explanation for the scale free behavior so prevalent on the Web
(Huberman 2001).

## Caching
* Computer scientists have gone beyond observations and explanations of Zipf’s law to apply it
to the design of content delivery on the Internet. A problem Internet service providers (ISP's)
face is devising ways to support rapidly growing web traffic while maintaining quality of
service in the form of fast response time for file requests. In order to quickly satisfy users'
request for web content, ISP's utilize caching, whereby frequently used files are copied and
stored “near” to users on the network. It is important to note, however, that the effectiveness
of caching relies heavily on the existence of Zipf’s law.
Let's say that there is a web server in the United States serving a page that is extremely
popular in a town in Europe. In the absence of caching, every time someone in that town
requests the page, their request travels across the Atlantic, reaches the US server, which in
turn sends the page back across the Atlantic to the requester in Europe.
To avoid sending unnecessary cross-Atlantic requests, the Internet service provider
serving the European town can place a proxy server near the town. The proxy server’s role is
to accept requests from the users and forward them on their behalf. Now, when the first user
requests the document, the request goes to the proxy. If the proxy cache does not contain the
document, it makes a request to the US server, which replies to the proxy. The proxy then
sends the file to the requesting user, and stores the file locally in a cache. When additional
users send their requests for the file to the proxy, the proxy can serve them the file directly
from its cache, without having to contact the webserver in the US. Of course, files that are
updated frequently, such as the front page of a news site, have an expiration time after which
the file is considered 'stale'. The cache uses the expiration time to determine when to request a
new version from the origin server.
Caching has two advantages. First, since the requests are served immediately from the
cache, the response time can be significantly faster than contacting the origin server. Second,
caching conserves bandwidth by avoiding redundant transfers along remote internet links. The
benefits of caching are confirmed by its wide use by ISPs. They benefit because they are able to reduce the amount of inter-ISP traffic that they have to pay for. Caching by proxies benefits
not only the ISPs and the users, but also the websites holding the original content. Their
content reaches the users more quickly and they avoid being overloaded themselves by two
many direct requests.
However, since any cache has a finite size, it is impossible for the cache to store all of the
files users are requesting. Here Zipf's law comes into play. Several studies (Cunha 1995,
Breslau 1999) have found the popularity of files requested follows a Zipf distribution. Hence,
the cache need only store the most frequently requested files in order to satisfy a large fraction
of users requests.

## Networks
The Internet is comprised of networks on many levels, and some of the most exciting con-
sequences of Zipf’s law have been discovered in this area. The World Wide Web is a network
of interconnected webpages and the Internet backbone is a physical network used to transmit
data, including web pages, in the form of packets, from one location to another.
Measurements on both the World Wide Web (Adamic 1999, Jeong 1999) and the Internet
backbone (Faloutsos 1999, Albert 2000) have shown that they differ significantly from the
classic Erdös-Rényi model of random graphs (Erdös 1960). While the traditional Erdös-Rényi
model has a Poisson node degree distribution, with most nodes having a characteristic number
of links, these networks approximately follow a Zipf or scale-free degree distribution p(k) ~
k−τ, where k is the node degree, and τ is the scale-free exponent. To account for these
observations, new random graph growth models have been developed that rely on the above
mentioned idea of preferential attachment (Albert 2002).
The scale free degree distribution of the Internet backbone, shown in Figure 3, implies
that some nodes in the network maintain a large number of connections (proportional to the
total size of the network), while for the most part nodes have just one or two connections.
This is a two edged sword when it comes to resilience of the network. It means that if a node
fails at random, it is most likely one with very few connections, and its failure won’t affect the
performance of the network overall. However, if one were to specifically target just a few of
the high degree nodes, the network could be adversely affected. Because many routes pass
through the high degree nodes, their removal would require rerouting through longer and less
optimal paths. Once a sufficient number of high degree nodes are removed, the network itself
can become fragmented, without a way to communicate from one location to another.
On a different level, one of the recent developments in the use of the Internet has been the
emergence of peer-to-peer (P2P) networks. These networks are used by millions of users daily
to exchange a variety of files directly with one another. Examples of P2P networks include
Napster, Gnutella, and Kazaa. Although Napster was immensely popular, it was forced to shut
down by the recording industry over concerns that users were trading copyrighted music files.
Part of the reason Napster could so easily be shut down is that it operated with a central
server. The users would report which files they were sharing to the central server, and when
they looked for additional files, they would query the central server to locate other users who
had those files.
4
10
In te rn e t
3
fit α = 0 .8 1

n u m b e r o f co n n e ctio
10

2
10

1
10

0
10

-1
10
0 1 2 3 4
10 10 10 ra n k 10 10

F igure 3. T he connectivity of the internet backbone at the autonom ous system (A S level).
E ach A S is itself a netw ork corresponding to a single ISP , business entity or educational
institution.



Having learned from Napster’s troubles, current peer-to-peer networks tend to be de-
centralized. That is, nodes connect directly to one another rather than to a central server. The
distribution in the number of computers a computer has connections to is a Zipf distribution
(recently it has shifted into a two-sided Zipf distribution, with a shallower exponent for the
high degree nodes and a steeper exponent for the low degree ones) (Ripeanu 2002). The
presence of Zipf’s law has implications for the search strategies used in P2P networks.
Currently, most P2P networks use a broadcast method of locating files. Because there is no
central server that queries can be sent to, each node broadcasts the query to all of its neighbors
who in turn broadcast to all of their neighbors, out to some fixed distance from the originating
node. As one can well imagine, the network can become quite congested with broadcasted
queries. Recent research has shown, however, that routing queries to the high degree nodes
may provide a degree of congestion relief, while maintaining a short response time (Adamic
2001). Again, knowledge of Zipf’s law in the connectivity distribution has offered a solution
to an Internet communication problem.
Finally, it has been shown that scale-free networks are more susceptible to viruses than
networks with a more even degree distribution. Namely, a virus spreading in a random
network needs to surpass a threshold of infectiousness in order not to die out. However, if the
network has a Zipf degree distribution, the virus can persist in the network indefinitely, no
matter what level of its infectiousness (Pastor-Satarros 2002).
Both email (Ebel 2002) and instant messaging networks (Smith 2002) have been shown to
be scale free. Some individuals have a large number of email contacts but most individuals
would keep only a few addresses in their contact lists. This wide variance in the connectivity
of electronic communication reflects the different degrees of communicativeness in people
and their different roles at work and in society overall. Over the past few years, email viruses
have plagued the Internet, no doubt facilitated by hubs, or individuals with large contact lists.
An email virus can be passed on as an attachment in email messages. Once the attachment is
opened, the virus can activate and cause the email program to send numerous infected emails
to email addresses from the person’s contact list. The “I love you” email virus alone infected
over 500,000 individual systems in May of 20002. Sometimes the sheer quantity of viral email can affect the Internet’s performance. But just as hubs (individuals or computers with
many contacts) can facilitate the spreading of a virus, they can also aid in preventing their
spread. Carefully im-munizing the hubs could stop the virus in its tracks.

##Conclusions

On the Internet, Zipf’s law appears to be the rule rather than the exception. It is present at the
level of routers transmitting data from one geographic location to another and in the content
of the World Wide Web. It is also present at the social and economic level, in how individuals
select the websites they visit and form peer-to-peer communities. The ubiquitous nature of
Zipf’s law in cyberspace has lead to a deeper understanding of Internet phenomena, and has
consequently influenced the way in which it has evolved.]

my boss and stakeholders are watching over my shoulder so make sure you do not omit anything and produce high quality markdown file

05_zipfs_law_and_the_internet.pdf

text
# Study Guide: Zipf’s Law and the Internet

**Coverage:** Processed **1 PDF file** and **8 pages** of article content corresponding to the attached PDF and the provided text.

## Table of Contents
- [Article Information](#article-information)
- [Abstract](#abstract)
- [Keywords](#keywords)
- [Introduction](#introduction)
- [Power Laws and Scale-Free Behavior](#power-laws-and-scale-free-behavior)
- [Figure 1: Web Measurement Distributions](#figure-1-web-measurement-distributions)
- [From PDF to Zipf Ranked Distribution](#from-pdf-to-zipf-ranked-distribution)
- [Figure 2 and Table 1: User Traffic and Popularity](#figure-2-and-table-1-user-traffic-and-popularity)
- [A Growth Model](#a-growth-model)
- [Caching](#caching)
- [Networks](#networks)
- [Peer-to-Peer Networks](#peer-to-peer-networks)
- [Viruses, Email, and Instant Messaging](#viruses-email-and-instant-messaging)
- [Conclusions](#conclusions)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [Glossary](#glossary)

## Article Information

### Title
- **Zipf’s law and the Internet**

### Authors
- **Lada A. Adamic**
- **Bernardo A. Huberman**

### Publication Information
- **Glottometrics 3, 2002, 143–150**

### Dedication
- *To honor G.K. Zipf*

### Correspondence Information
- Address correspondence to: **Lada A. Adamic, HP Laboratories, 1501 Page Mill Road, ms 1139, Palo Alto, CA 94304, USA**
- Email: **ladamic@exch.hpl.hp.com**

## Abstract

- **Zipf's law** governs many features of the Internet.
- Observations of Zipf distributions, while interesting in themselves, have strong implications for the design and function of the Internet.
- The connectivity of Internet routers influences network robustness.
- The distribution in the number of email contacts affects the spread of email viruses.
- Web caching strategies are also formulated to account for a Zipf distribution in requests for webpages.

## Keywords

- **Zipf´s law**
- **caching**
- **networks**

## Introduction

- The wide adoption of the Internet fundamentally altered how people communicate, gather information, conduct business, and make purchases.
- As use of the World Wide Web and email skyrocketed, computer scientists and physicists rushed to characterize the phenomenon.
- They discovered a widespread pattern in Internet measurements: there are many small elements on the Web, but few large ones.

### Examples Given
- A few sites consist of **millions of pages**, but millions of sites contain only a handful of pages.
- Few sites contain **millions of links**, but many have one or two.
- Millions of users flock to a few select sites, while millions of others receive little attention.

### Historical Context
- This pattern had already been observed in:
  - **Income distributions** (*Pareto 1896*)
  - **Word frequencies in text** (*Zipf 1932*)
  - **City sizes** (*Zipf 1949*)

## Power Laws and Scale-Free Behavior

### Power Law Definition
- The article states that the probability of attaining a certain size $x$ is proportional to $x^{-\tau}$, where $\tau \geq 1$.

### Key Characteristics
- Unlike a Gaussian distribution, a **power law distribution** has no typical scale.
- For this reason, it is often called **scale-free**.
- A power law gives a finite probability to very large elements.
- By contrast, the exponential tail of a Gaussian makes values much larger than the mean extremely unlikely.

### Example Contrast
- **City sizes** follow a power law and therefore include a few mega cities that are orders of magnitude larger than the mean city size.
- **Human heights**, which are closer to a Gaussian distribution, do not permit a person several times taller than average.

## Figure 1: Web Measurement Distributions

### Description
- **Figure 1** shows a series of scale-free distributions in website size measured by:
  - Number of pages
  - Number of links given to other sites
  - Number of links received from other sites
  - Number of unique users visiting the site

### Subfigures
- **a) Number of Pages**
  - Measured using crawls by **infoseek.com** and **archive.org**
- **b) Number of Visitors**
  - Binned distribution with a power-law fit of **$\tau = -2.07$**
- **c) Out-link Distribution**
  - Number of links pointing **from** a site
  - Fit reported as **$\tau = 2.0$**
- **d) In-link Distribution**
  - Number of links pointing **to** a site
  - Fit reported as **$\tau = 2.0$**

### Figure Note
- The distributions were measured in **1997**.
- The scales span broad ranges such as **$10^0$ to $10^6$** in the reformatted excerpt.

> NOTE: The provided text reports the visitor power-law fit in Figure 1b as **$\tau = -2.07$**. This is preserved exactly as given in the source material, even though surrounding notation elsewhere in the article describes power-law exponents in forms that are typically presented differently.

## From PDF to Zipf Ranked Distribution

### Motivation
- The article explains that the distributions in Figure 1 are plotted as a **probability density function (PDF)**.
- These can also be recast as a **Zipf ranked distribution**.
- The PDF emphasizes the count of small elements.
- The ranked distribution emphasizes the size of the largest ones.

### PDF Formula
$$
p(x) = Cx^{-\tau}
$$

### Probability of Size $y$ or Larger
$$
P(x > y) = \sum_{y}^{\infty} Cx^{-\tau} \approx Ay^{-\tau+1}
$$

- Here, **$C$** and **$A$** are constants.

### Expected Rank
- If there are **$N$ total websites**, the expected number of sites greater than a threshold is given in the provided reformatted text as:

$$
r = NAy^{\tau-1}
$$

> NOTE: The PDF text extraction and the reformatted excerpt present the rank relationship in a compact algebraic form that may reflect formatting loss in the original typesetting. It is reproduced here exactly from the provided source text.

### Ranked Variable Size
- Solving for $y$, the size of the **$r^{th}$ ranked variable** is proportional to:

$$
r^{-1/(\tau-1)} = r^{-\alpha}
$$

- **$\alpha$** is the **Zipf rank exponent**.

### Interpretation
- A pure power-law PDF yields a Zipf ranked distribution.
- The ranked form provides a convenient way to study the concentration of the largest websites or most popular sites.

## Figure 2 and Table 1: User Traffic and Popularity

### Figure 2
- **Figure 2** shows sites ranked by the number of **unique AOL visitors** received on **December 1, 1997**.
- AOL is identified as the largest Internet service provider in the United States at that time.
- The fit is described as a **Zipf distribution**:

$$
n_r \sim r^{-1}
$$

### Notes on Shape
- The distribution shows **mild concavity**.
- The ranked exponent is **1**.

### Table 1
- **Table 1** gives the distribution of user volume among:
  - General sites
  - Adult sites
  - `.edu` domain sites

- It is based on counting the number of **unique AOL visitors** on **December 1, 1997**.

| % of Sites | % User Volume |
|---|---|
| 0.1% | 32.36% |
| 1% | 55.63% |
| 5% | 74.81% |
| 10% | 82.26% |
| 50% | 94.92% |

### Key Observation
- A small number of sites, such as **Yahoo**, are extremely popular and capture a disproportionate amount of traffic.

> NOTE: The provided user text states that “the bottom 50% of sites share roughly 5% of the total volume,” while the table states that the **top 50% of sites account for 94.92%** of user volume. These statements are broadly consistent as complements, but the latter is the directly tabulated form.

## A Growth Model

### Main Claim
- The article proposes an intuitive **growth model** to explain the pervasiveness of Zipf distributions on the Internet.

### Scope
- The argument is formulated in terms of the **number of web pages hosted on a website**.
- Similar reasoning can also be applied to:
  - Number of visitors
  - Number of links

### Assumption 1: Proportional Growth / Preferential Attachment
- The number of pages added to or removed from a site is proportional to the number of pages already present.
- Example:
  - A site with **a million pages** may have a team of webmasters or automatically generated content.
  - It can easily gain or lose several thousand pages in a day.
  - A site with only a handful of pages is much less likely to suddenly add a thousand more.

### Intermediate Result
- This **multiplicative stochastic growth process** yields a **lognormal distribution** in the number of pages at a site after a fixed period of time.

### Additional Ingredient: Exponential Website Growth
- The World Wide Web is not fixed.
- During its first decade, sites appeared at an **exponential rate**.
- An **exponentially weighted mixture of lognormals** produces a **power-law distribution exactly**.

### Missing Ingredient
- If growth depended only on time, older sites should generally be larger than recently founded ones.
- However, studies found only a **weak correlation** between site size and site age.
- Some very popular sites were founded later, while some early sites did not become highly popular.

### Additional Assumption: Variable Growth Rates
- Sites can grow at different rates depending on:
  - Type of content
  - Interest generated by that content

### Consequence
- Incorporating variability in growth rates again yields **power-law distributions** with varying exponents.
- The greater the difference in growth rates among sites, the lower the exponent **$\tau$**.
- A lower **$\tau$** implies **greater inequality** in site sizes.

### Summary of the Model
- **Stochastic multiplicative growth**
- Combined with:
  - Different site birth times, and/or
  - Different site growth rates
- Leads to an explanation for the **scale-free behavior** prevalent on the Web.

## Caching

### Main Point
- Computer scientists have not only observed and explained Zipf’s law; they have also applied it to **Internet content delivery design**.

### Problem Faced by ISPs
- Internet service providers must support rapidly growing web traffic while maintaining quality of service, especially **fast response times** for file requests.

### What Caching Does
- ISPs use **caching**, in which frequently used files are copied and stored **near users** on the network.
- The effectiveness of caching relies heavily on the existence of **Zipf’s law**.

### Cross-Atlantic Example
- Suppose a web server in the United States serves a page that is extremely popular in a town in Europe.
- Without caching:
  - Each request travels across the Atlantic to the US server.
  - The server sends the page back across the Atlantic to the requester.

### Proxy Server Example
- To reduce unnecessary long-distance traffic, the ISP can place a **proxy server** near the European town.
- The proxy:
  - Accepts user requests
  - Forwards them on users’ behalf if needed
  - Stores returned files locally in a cache

### Request Flow
1. A first user requests a document.
2. The request goes to the proxy.
3. If the proxy does not already have the file, it requests the file from the US server.
4. The US server sends the file to the proxy.
5. The proxy sends the file to the user and stores it locally.
6. Later users can receive the file directly from the proxy cache.

### Staleness
- Files that change often, such as a news homepage, have an **expiration time**.
- After expiration, the file is considered **stale**.
- The cache uses the expiration time to decide when to fetch a new version from the origin server.

### Advantages of Caching
- **Faster response time**, because requests are served from a nearby cache.
- **Bandwidth conservation**, because redundant remote transfers are avoided.

### Who Benefits
- **ISPs**, because they reduce inter-ISP traffic they may have to pay for.
- **Users**, because content arrives more quickly.
- **Origin websites**, because they avoid being overloaded by too many direct requests.

### Finite Cache Size
- A cache has finite size and cannot store every requested file.
- This is where Zipf’s law becomes critical.

### Zipf and Caching Strategy
- Studies including **Cunha 1995** and **Breslau 1999** found that file popularity follows a **Zipf distribution**.
- Therefore, the cache need only store the **most frequently requested files** to satisfy a large fraction of user requests.

## Networks

### Internet as Multi-Level Networks
- The Internet consists of networks on many levels.
- Zipf’s law has important consequences in this area.

### Two Examples
- The **World Wide Web** is a network of interconnected webpages.
- The **Internet backbone** is a physical network that transmits packets, including web pages, from one location to another.

### Comparison to Erdös-Rényi Random Graphs
- Measurements on both:
  - The World Wide Web (*Adamic 1999; Jeong 1999*)
  - The Internet backbone (*Faloutsos 1999; Albert 2000*)
- Show significant differences from the classic **Erdös-Rényi** random graph model (*Erdös 1960*).

### Degree Distributions
- In the traditional Erdös-Rényi model, node degree follows a **Poisson distribution**.
- Most nodes have a characteristic number of links.
- In contrast, Internet-related networks approximately follow a **Zipf** or **scale-free degree distribution**:

$$
p(k) \sim k^{-\tau}
$$

- Here:
  - **$k$** is the node degree
  - **$\tau$** is the scale-free exponent

### Modeling Implication
- To explain these observations, new random graph growth models were developed based on **preferential attachment**.

### Internet Backbone Resilience
- The scale-free degree distribution of the Internet backbone implies:
  - Some nodes maintain a very large number of connections, proportional to network size
  - Most nodes have only one or two connections

### Two-Edged Sword
- This structure has mixed implications for network resilience.

#### Random Failure
- If a node fails at random, it is most likely a low-degree node.
- Its failure is unlikely to affect overall network performance.

#### Targeted Attack
- If high-degree nodes are specifically targeted:
  - The network may be adversely affected
  - Many routes must be rerouted through longer, less optimal paths
  - If enough high-degree nodes are removed, the network can fragment and lose global connectivity

### Figure 3
- **Figure 3** shows the connectivity of the Internet backbone at the **autonomous system (AS) level**.
- Each AS is itself a network corresponding to:
  - A single ISP
  - A business entity
  - An educational institution
- The fit reported in the figure is **$\alpha = 0.81$**.

## Peer-to-Peer Networks

### Emergence of P2P
- A major development in Internet use was the emergence of **peer-to-peer (P2P) networks**.
- Millions of users use these networks daily to exchange files directly.

### Examples
- **Napster**
- **Gnutella**
- **Kazaa**

### Napster’s Centralized Design
- Napster had a **central server**.
- Users reported which files they were sharing to the central server.
- When searching for additional files, users queried that server to locate others with the files.

### Shutdown Context
- Napster was forced to shut down by the recording industry because of concerns over trading copyrighted music files.
- Its centralized architecture made it easier to shut down.

### Decentralized P2P
- Later P2P networks tended to be **decentralized**.
- Nodes connect directly to one another rather than to a central server.

### Connectivity Distribution
- The number of connections per computer follows a **Zipf distribution**.
- The article notes a more recent shift to a **two-sided Zipf distribution**:
  - Shallower exponent for high-degree nodes
  - Steeper exponent for low-degree nodes

### Search Strategy Implications
- Many P2P networks use a **broadcast method** to locate files.
- Each node forwards the query to all neighbors, who forward to all of their neighbors, out to a fixed distance.
- This can cause significant congestion.

### High-Degree Routing
- Research showed that routing queries to **high-degree nodes** can relieve congestion while maintaining short response time.
- This is presented as another example of using knowledge of Zipf’s law to solve an Internet communication problem.

## Viruses, Email, and Instant Messaging

### Virus Persistence in Scale-Free Networks
- Scale-free networks are more susceptible to viruses than networks with more even degree distributions.
- In a random network, a virus must exceed an infectiousness threshold in order not to die out.
- In a network with a Zipf degree distribution, a virus can persist indefinitely regardless of its infectiousness level.

### Email and Instant Messaging
- Both **email networks** (*Ebel 2002*) and **instant messaging networks** (*Smith 2002*) are described as **scale free**.

### Social Interpretation
- Some individuals maintain many email contacts, while most maintain only a few.
- This reflects differences in:
  - Communicativeness
  - Roles at work
  - Roles in society overall

### Email Viruses
- Email viruses have plagued the Internet, aided by **hubs** with large contact lists.
- A virus can spread as an email attachment.
- Once opened, it can cause the email program to send infected messages to many addresses in the user’s contact list.

### Example
- The **“I love you”** email virus infected over **500,000 individual systems** in **May 2000**.

### Performance Impact
- Large quantities of viral email can degrade Internet performance.

### Immunizing Hubs
- The same hubs that help viruses spread can also help stop them.
- Carefully immunizing hubs could halt virus propagation.

## Conclusions

- On the Internet, **Zipf’s law appears to be the rule rather than the exception**.
- It appears:
  - At the level of routers transmitting data geographically
  - In the content of the World Wide Web
  - At the social and economic level, in website selection and peer-to-peer communities
- The ubiquity of Zipf’s law in cyberspace has led to a deeper understanding of Internet phenomena.
- It has consequently influenced how the Internet has evolved.

## Glossary

### **Autonomous System (AS)**
- A network corresponding to a single ISP, business entity, or educational institution in the Internet backbone figure.

### **Caching**
- A strategy in which frequently requested files are copied and stored near users to improve response time and reduce bandwidth usage.

### **Degree Distribution**
- The distribution of the number of links or connections associated with nodes in a network.

### **Erdös-Rényi Model**
- A classic random graph model with a Poisson node degree distribution.

### **Gaussian Distribution**
- A distribution with a characteristic scale and an exponential tail, used in the article as a contrast to power laws.

### **Lognormal Distribution**
- The distribution produced by multiplicative stochastic growth over a fixed period of time.

### **Peer-to-Peer (P2P) Network**
- A network in which users exchange files directly with one another rather than relying on a central server.

### **Poisson Distribution**
- The node degree distribution associated with the traditional Erdös-Rényi random graph model.

### **Preferential Attachment**
- A growth mechanism in which the number of additions to a site or node is proportional to its current size or degree.

### **Proxy Server**
- A server near users that receives requests, forwards them if necessary, and stores returned files in a local cache.

### **Scale-Free**
- A property of a distribution with no typical scale, commonly associated with power laws.

### **Stale File**
- A cached file whose expiration time has passed and which must be refreshed from the origin server.

### **Zipf Distribution**
- A ranked distribution in which size or frequency falls off approximately as a power of rank.

### **Zipf Rank Exponent**
- The exponent **$\alpha$** appearing in the rank-size relationship $r^{-\alpha}$.

### **Zipf’s Law**
- A power-law-like regularity governing ranked frequencies or sizes, described in the article as pervasive across many Internet phenomena.
---


