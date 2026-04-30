# Week03 Quiz

**Q1**
**1 Point**

Which of the following are assumptions of the distributional hypothesis?
- ( ) The meaning of a word can be inferred from the contexts in which it appears.
- ( ) The distributional information of a word can be captured by co-occurrence matrices.
- ( ) The meaning of a word is fixed and does not change over time.
- ( ) Words that appear in similar contexts tend to have similar meanings.

**Q2**
**1 Point**

Which of the following factors affect the quality of word embeddings based on the distributional hypothesis?
- ( ) The part-of-speech tags of the words in the corpus
- ( ) The dimensionality of the vector space used to represent the embeddings
- ( ) The choice of window size used to define word context
- ( ) The size of the corpus used to generate the embeddings

**Q3**
**1 Point**

What is the difference between the CBOW and Skip-gram models?

*(Text box for answer)*

**Q4**
**1 Point**

What is vector semantics?
- ( ) A way to represent words as strings of characters
- ( ) A way to represent words as boolean values
- ( ) A way to represent words as numerical vectors
- ( ) A way to represent words as images

**Q5**
**1 Point**

Which of the following is an example of an arithmetic operation that can be performed using the parallelogram property of word embeddings?
- ( ) happy + sad = emotion
- ( ) cat + dog = animal
- ( ) king - queen + man = woman
- ( ) red + blue = green

**Q6**
**1 Point**

What is this code trying to do?

*(Text box for answer)*

```python
import gensim.downloader as api
from sklearn.cluster import KMeans

# Load the pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

# Get the word vectors for a list of words
words = ['cat', 'dog', 'fish', 'bird', 'snake', 'hamster', 'rabbit']
vectors = [model[word] for word in words]

# Perform KMeans clustering on the vectors
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(vectors)

# Print the words in each cluster
for i in range(3):
    print('Cluster {}:'.format(i+1))
    for j in range(len(clusters)):
        if clusters[j] == i:
            print('  ', words[j])
```

**Q7**
**1 Point**

Which of the following methods can be used to compute co-occurrence matrices for word embeddings?
- ( ) Pointwise Mutual Information (PMI)
- ( ) Singular Value Decomposition (SVD)
- ( ) Word2Vec

**Q8**
**1 Point**

Which of the following algorithms is used for training the Skip-gram model?
- ( ) K-Nearest Neighbors
- ( ) Gradient Descent
- ( ) Naive Bayes
- ( ) Support Vector Machines

**Q9**
**1 Point**

What is a co-occurrence matrix?

*(Text box for answer)*

**Q10**
**1 Point**

What is the objective of the Skip-gram model during training?
- ( ) To maximize the probability of predicting a context word given a target word
- ( ) To minimize the loss function
- ( ) To minimize the distance between word embeddings
- ( ) To maximize the dot product between word embeddings

**Q11**
**1 Point**

What is the role of the negative sampling technique in the Skip-gram model?

*(Text box for answer)*

**Q12**
**1 Point**

What is the distributional hypothesis in NLP?

*(Text box for answer)*
