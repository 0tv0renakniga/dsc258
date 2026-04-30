# Module 2 Quiz

You’re not a pain at all — this is exactly how you actually learn it. Here’s a simple, quiz-focused way to solve each one.

## Q1 F1 score

### What to do
1. Read the confusion matrix and identify:
- **TP** = predicted positive and actually positive = **850**
- **FP** = predicted positive and actually negative = **10**
- **FN** = predicted negative and actually positive = **50**
- **TN** = predicted negative and actually negative = **90** [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

2. Use the F1 formula:
\[
F1=\frac{2TP}{2TP+FP+FN}
\]
 [telnyx](https://telnyx.com/learn-ai/calculating-f1-score)

3. Plug in the values:
\[
F1=\frac{2(850)}{2(850)+10+50}
\]

4. Simplify:
\[
F1=\frac{1700}{1760}\approx 0.9659
\]

5. Pick the matching answer:
- **0.9659** [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

### Fast shortcut
For confusion-matrix questions:
- **Ignore TN**
- Use only **TP, FP, FN** for F1 [labelyourdata](https://labelyourdata.com/articles/machine-learning/confusion-matrix)

***

## Q2 Bag-of-words classification

### What to do
Go through each statement and ask: **always true, usually true, or false?**

1. **“More complicated TF-IDF variants are always better than simple variants.”**
- False.
- In ML, “more complicated” does **not** mean “always better.” Performance depends on the dataset and task. [coursera](https://www.coursera.org/articles/what-is-tfidf)

2. **“TF gives more weights to popular words.”**
- True.
- TF increases when a word appears more often in a document. [semrush](https://www.semrush.com/blog/tf-idf/)

3. **“IDF gives more weights to infrequent words.”**
- True.
- IDF is higher for words that appear in fewer documents. [vishwasg](https://vishwasg.dev/blog/2025/01/01/a-comprehensive-guide-on-tf-idf/)

4. **“TF-IDF based models must have better performance than binary vectorization.”**
- False.
- “Must” makes it too strong. Sometimes binary works better. [coursera](https://www.coursera.org/articles/what-is-tfidf)

### Correct boxes to check
- [x] **TF gives more weights to popular words** [semrush](https://www.semrush.com/blog/tf-idf/)
- [x] **IDF gives more weights to infrequent words** [coursera](https://www.coursera.org/articles/what-is-tfidf)

### Fast shortcut
On “choose all that apply”:
- Watch for extreme words like **always**, **must**, **never** — those are often false. [coursera](https://www.coursera.org/articles/what-is-tfidf)

***

## Q3 Validation set

### What to do
1. Think about the roles:
- **Train set** = fit the model
- **Validation set** = try things out / tune
- **Test set** = final check [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/training-vs-testing-vs-validation-sets/)

2. Ask: why not just use the test set for tuning?
- Because then you start overfitting to the test set. [codecademy](https://www.codecademy.com/article/training-validation-test-set)

3. Write the main advantage:
- A validation set lets you do a **practice run** before final testing. [community.deeplearning](https://community.deeplearning.ai/t/why-do-we-need-to-have-a-validation-set-for-training/267572)

### Simple answer you can use
> A validation set gives us a practice run to tune the model before using the test set. It helps us detect overfitting, such as memorizing the training data, and keeps the test set for final unbiased evaluation. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/training-vs-testing-vs-validation-sets/)

### Fast shortcut
If they ask the advantage of a validation set, say:
- **tune model**
- **detect overfitting**
- **keep test set unbiased** [codecademy](https://www.codecademy.com/article/training-validation-test-set)

***

## Q4 TF-IDF

### What to do
1. Recall the basic idea:
\[
TF\text{-}IDF = TF \times IDF
\]
 [semrush](https://www.semrush.com/blog/tf-idf/)

2. Find **TF**:
- The word “hello” appears **K** times in a document with **T** words.
- So term frequency is roughly:
\[
TF=\frac{K}{T}
\]
 [semrush](https://www.semrush.com/blog/tf-idf/)

3. Find **IDF**:
- “hello” appears in about \(1/5\) of all documents.
- That means document frequency is about \(N/5\).
- So:
\[
IDF=\log\left(\frac{N}{N/5}\right)=\log(5)
\]
 [vishwasg](https://vishwasg.dev/blog/2025/01/01/a-comprehensive-guide-on-tf-idf/)

4. Multiply TF and IDF:
\[
TF\text{-}IDF=\frac{K}{T}\log(5)
\]

5. Match to the answer choice:
- **\(K * \log(5) / T\)** [semrush](https://www.semrush.com/blog/tf-idf/)

### Fast shortcut
For TF-IDF problems:
- TF = **how often in this document**
- IDF = **how rare across all documents**
- Then multiply them [coursera](https://www.coursera.org/articles/what-is-tfidf)

***

## One-page cheat sheet

| Question type | Steps |
|---|---|
| F1 from confusion matrix | Find TP, FP, FN, then use \(\frac{2TP}{2TP+FP+FN}\)  [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) |
| TF / IDF true-false | TF = favors frequent-in-document words, IDF = favors rare-across-corpus words  [semrush](https://www.semrush.com/blog/tf-idf/) |
| Validation set | Say it helps tune model, detect overfitting, and preserve test set for final evaluation  [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/training-vs-testing-vs-validation-sets/) |
| TF-IDF formula | Compute TF, compute IDF, multiply them  [semrush](https://www.semrush.com/blog/tf-idf/) |

## Final answers

- **Q1:** 0.9659 [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- **Q2:** TF gives more weights to popular words; IDF gives more weights to infrequent words. [semrush](https://www.semrush.com/blog/tf-idf/)
- **Q3:** Validation set helps tune the model, detect overfitting, and keep the test set unbiased for final evaluation. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/training-vs-testing-vs-validation-sets/)
- **Q4:** \(K * \log(5) / T\) [coursera](https://www.coursera.org/articles/what-is-tfidf)

If you want, I can make this into an even more compact **“how to recognize the right answer fast”** version for quiz use.
