==================================================
WEEK 05 STUDY GUIDE: NEURAL LANGUAGE MODELS
QUIZ-TARGETED, LECTURE-GROUNDED, NO EXTERNAL CITATIONS
==================================================

This guide is rewritten to match the testing style seen in earlier quizzes:
- definition plus purpose,
- comparison across related methods,
- why an assumption is useful,
- tradeoffs and limitations,
- short-answer ready phrasing.

--------------------------------------------------
0. BIG PICTURE
--------------------------------------------------

Week 05 transitions from count-based language models to neural language models.

Core progression:
- Classical n-gram LM: fixed recent context, count-based probabilities.
- Neural n-gram LM: still fixed context, but neural parameterization.
- RNN LM: recurrent hidden state, full left-context factorization.

Main high-yield ideas:
- Markov assumption vs chain rule.
- Why full context is desirable but hard (parameter explosion, sparsity).
- Neural parameter sharing as a response to sparsity.
- Neural n-gram structure and input features.
- RNN recurrence, parameters, and output probabilities.
- Training objective = maximize log likelihood.
- Sampling = generate one word at a time.

--------------------------------------------------
1. FOUNDATIONAL CONTRAST: MARKOV ASSUMPTION VS CHAIN RULE
--------------------------------------------------

Markov assumption:
- Predict next word using only a limited recent window of k previous words.
- Formula:
  $$p(x_1, ..., x_n) = \prod_i p(x_i | x_{i-k}, ..., x_{i-1})$$

Chain rule (no assumption):
- Predict next word using the entire left context.
- Formula:
  $$p(x_1, ..., x_n) = \prod_i p(x_i | x_1, ..., x_{i-1})$$

Why the lecture cares:
- Full left context can capture long-range interactions between words.
- But conditioning on the entire left context is difficult if done naively.
- Naive full conditioning leads to too many parameters and sparsity.

Quiz-ready statement:
- The Markov assumption simplifies learning by restricting context, while the full chain rule is more expressive but harder to parameterize directly.

Typical trap:
- Do not say the full-context model is automatically better in practice. The lecture says naive full conditioning causes parameter and sparsity problems.

--------------------------------------------------
2. WHY NEURAL LANGUAGE MODELS
--------------------------------------------------

Core idea:
- Use neural networks to parameterize the conditional distribution on the next word.

Why this helps:
- Neural architectures allow parameter sharing between many different conditional next-word distributions.
- This can result in fewer parameters overall and can help the model deal with sparsity.

Quiz-ready statement:
- Neural language models help by sharing parameters across contexts instead of treating every context as a completely separate probability table.

Typical trap:
- Do not claim neural models eliminate sparsity entirely. The lecture-supported claim is that they can help with sparsity through parameter sharing.

--------------------------------------------------
3. NEURAL N-GRAM LANGUAGE MODELS
--------------------------------------------------

Definition:
- Still uses a fixed-size context window (Markov-style restriction).
- Predicts:
  $$p(x_i | x_{i-k}, ..., x_{i-1})$$

Parameterizations shown in lecture:
- Logistic regression form:
  $$p(x_i | x_{i-k}, ..., x_{i-1}) \propto \exp(w^T f(x_{i-k}, ..., x_{i-1}))$$
- Fully connected neural network form:
  $$p(x_i | x_{i-k}, ..., x_{i-1}) \propto \exp(w_3^T \sigma(W_2^T \sigma(W_1^T f(x_{i-k}, ..., x_{i-1}))))$$

Hidden-layer decomposition:
- First hidden layer: $h_1 = \sigma(W_1^T f)$
- Second hidden layer: $h_2 = \sigma(W_2^T h_1)$
- Output layer: $p(x_i | ...) \propto \exp(w_3^T h_2)$

Input features:
- Feature vector is concatenated embeddings of context words:
  $$f(x_{i-k}, ..., x_{i-1}) = \mathrm{concat}(e(x_{i-k}), ..., e(x_{i-1}))$$

Training objective:
- Maximize sum of log conditional probabilities:
  $$\max_\theta \sum_i \log p_\theta(x_i | x_{i-k}, ..., x_{i-1})$$

Optimization:
- Gradient ascent (or stochastic gradient ascent).

What changed vs classical n-grams:
- Classical: count-based estimation.
- Neural: embeddings and neural-network weights.

What did not change:
- The context remains fixed-size.

Quiz-ready one-sentence answer:
- A neural n-gram model predicts the next word from a fixed context window using concatenated context embeddings and a feed-forward neural network.

Typical traps:
- Neural n-gram does not remove the fixed context limit.
- Neural n-gram is not recurrent.
- The input is the context feature vector, not just a single word.

--------------------------------------------------
4. RNN LANGUAGE MODELS
--------------------------------------------------

Concept:
- RNN LMs break the fixed-window restriction and model prediction using the full left context in the factorization.

Sequence factorization:
- $$p(x_1, ..., x_n) = \prod_i p(x_i | x_1, ..., x_{i-1})$$

Hidden-state update (recurrence):
- $$h_i = \sigma(W^T h_{i-1} + U^T x_i)$$
- $h_{i-1}$: previous hidden state (summary of $x_1...x_{i-1}$)
- $x_i$: input representation (embedding) of current word
- $W$: hidden-to-hidden weights
- $U$: input-to-hidden weights
- $\sigma$: nonlinearity (e.g., sigmoid)

Interpretation:
- The hidden state summarizes the context seen so far.

Output distribution:
- Next-word probability given history $x_1...x_i$:
  $$p(x_{i+1} | x_1,...,x_i) = \mathrm{softmax}(V h_i)$$
- For a specific candidate word:
  $$p(x_4 = cat | x_1, x_2, x_3) = \frac{\exp(v(cat) \cdot h_3)}{\sum_x \exp(v(x) \cdot h_3)}$$

Parameters (explicit from lecture):
- $$\theta = (E, V, U, W, h_0)$$
- $E$: input embedding matrix
- $V$: output parameter matrix
- $U$: input-to-hidden parameters
- $W$: hidden-to-hidden parameters
- $h_0$: initial hidden state

Quiz-ready one-sentence answer:
- An RNN language model uses a recurrent hidden state to summarize previous words and predict the next word using a softmax distribution over the vocabulary.

Typical traps:
- "Break the Markov assumption" does not mean the model stores every full history explicitly. The model represents that context through hidden states.

--------------------------------------------------
5. NEURAL N-GRAM VS RNN (HIGH-YIELD COMPARISON)
--------------------------------------------------

| Aspect                     | Neural n-gram                              | RNN language model                          |
|----------------------------|--------------------------------------------|---------------------------------------------|
| Context used               | Fixed recent window                        | Full left context in factorization         |
| Core mechanism             | Feed-forward network                       | Recurrent hidden state                      |
| Input representation       | Concatenated context embeddings            | Current embedding + previous hidden state  |
| Main benefit               | Shared neural parameterization for fixed context | Context summary not tied to a fixed window |
| Main limitation            | Still fixed context (cannot capture long dependencies) | Training difficulties are not detailed in lecture |

Note: Vanishing gradients are a known issue but not explicitly discussed in the provided lecture slides. For strict lecture-grounding, avoid claiming them unless the text includes it.

--------------------------------------------------
6. SAMPLING (GENERATION)
--------------------------------------------------

Procedure (autoregressive generation):
- First word: $x_1 \sim p(x_1)$
- Second word: $x_2 \sim p(x_2 | x_1)$
- Third word: $x_3 \sim p(x_3 | x_1, x_2)$
- In general: $x_i \sim p(x_i | x_1, ..., x_{i-1})$

Quiz-ready statement:
- Text generation in an autoregressive language model proceeds by repeatedly sampling the next word conditioned on the words generated so far.

Typical trap:
- Do not confuse sampling with training. Sampling uses learned conditional distributions to generate text; training adjusts parameters to improve likelihood on observed data.

--------------------------------------------------
7. TRAINING RNN LANGUAGE MODELS
--------------------------------------------------

Single-sentence log likelihood:
- $$\log p(x_1, ..., x_n) = \sum_i \log p(x_i | x_1, ..., x_{i-1})$$

Corpus log likelihood (multiple sentences):
- $$\sum_k \log p(x_1^{(k)}, ..., x_n^{(k)}) = \sum_k \sum_i \log p(x_i^{(k)} | x_1^{(k)}, ..., x_{i-1}^{(k)})$$

Training objective:
- $$\max_\theta \sum_k \log p(x_1^{(k)}, ..., x_n^{(k)}; \theta)$$

Gradient ascent update:
- $$\theta = \theta + \eta \cdot \nabla_\theta \sum_k \log p(x_1^{(k)}, ..., x_n^{(k)}; \theta)$$

Quiz-ready statement:
- RNN language models are trained by maximizing corpus log likelihood using gradient ascent.

Typical trap:
- The probability of a sequence is a product of conditionals, but the training objective is written as a sum of log probabilities (log likelihood).

--------------------------------------------------
8. EXPERIMENTS SECTION (WHAT LECTURE PROVIDES)
--------------------------------------------------

What is explicitly supported:
- The experiments lecture contains graphs related to recurrent language models.
- It also contains generated samples from recurrent language models (e.g., Shakespeare-like text).

What is not supported:
- The extracted slides do not include readable numeric conclusions or exact performance comparisons (e.g., "perplexity improves by X").

Safe statement:
- The experiments section illustrates recurrent language models with graphs and generated text samples, but the extracted material alone does not support detailed quantitative conclusions.

--------------------------------------------------
9. CROSS-WEEK REVIEW REMINDERS (FROM EARLIER COURSE MATERIAL)
--------------------------------------------------

These ideas are useful for quizzes and connect naturally to week05, but they come from earlier lectures (e.g., Week 04). They are not claimed as new week05 content.

- Held-out (validation) data is used for tuning hyperparameters.
- Test data is for final evaluation.
- Training-set log likelihood tends to overestimate future unseen-data performance.

These are good to remember because the instructor has already tested them.

--------------------------------------------------
10. MOST LIKELY QUIZ QUESTIONS (PREDICTED FROM PATTERN)
--------------------------------------------------

1. Which model still uses a fixed-size context window?
   - Neural n-gram model (and classical n-gram).

2. Which model uses the full left context in the factorization?
   - RNN language model.

3. What is the difference between the Markov assumption and the chain rule?
   - Markov assumption uses a fixed recent window; chain rule conditions on all previous words.

4. Why use neural language models?
   - They allow shared parameterization of next-word distributions and can help with sparsity.

5. What is the input feature vector in a neural n-gram model?
   - The concatenation of context word embeddings.

6. What is the training objective of the neural n-gram model?
   - Maximize the sum of log conditional probabilities.

7. What are the parameters of the RNN language model?
   - $\theta = (E, V, U, W, h_0)$.

8. What does the hidden state represent in an RNN LM?
   - A summary of the previous context (all prior words).

9. How is text generated from an RNN language model?
   - By sampling one word at a time from the learned next-word conditional distributions.

10. What does the experiments section clearly show?
    - Graphs and generated samples for recurrent language models.

--------------------------------------------------
11. ULTRA-CONDENSED EXAM SHEET
--------------------------------------------------

- Markov assumption: $p(x_1, ..., x_n) = \prod_i p(x_i | x_{i-k}, ..., x_{i-1})$
- Chain rule: $p(x_1, ..., x_n) = \prod_i p(x_i | x_1, ..., x_{i-1})$
- Why neural models: shared parameterization, fewer parameters overall, helps sparsity.
- Neural n-gram: fixed context + concatenated embeddings + feed-forward network.
- Neural n-gram objective: $\max_\theta \sum_i \log p_\theta(x_i | x_{i-k}, ..., x_{i-1})$
- RNN LM: recurrent hidden state summarizes context.
- RNN recurrence: $h_i = \sigma(W^T h_{i-1} + U^T x_i)$
- RNN output: $\mathrm{softmax}(V h_i)$
- RNN parameters: $(E, V, U, W, h_0)$
- Training: maximize summed log likelihood with gradient ascent.
- Sampling: $x_i \sim p(x_i | x_1, ..., x_{i-1})$