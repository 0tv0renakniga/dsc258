# Module 4 Quiz

## Q1 (1 Point)
Suppose you estimate a 4-gram model on a training dataset using maximum likelihood (i.e. count-and-divide). And now, instead of using a separate test dataset, you evaluate the trained model's log likelihood on that same training dataset.
This evaluated log likelihood will (in expectation) be ______ of the log likelihood the model will achieve on future data that was not part of train.
( ) an underestimate
( ) an overestimate
( ) an accurate estimate

## Q2 (1 Point)
Which type of smoothing method uses a weighted combination of different orders of N-gram model, each separately estimated using maximum likelihood?
( ) absolute discounting
( ) linear interpolation
( ) Kneser-Ney

## Q3 (1 Point)
The number of parameters in the conditional probability tables that comprise an N-gram model grow ______ with $N$.
( ) exponentially
( ) linearly
( ) quadratically

## Q4 (1 Point)
If a smoothing method has smoothing hyper-parameters (e.g. the interpolation weights $\lambda$ used in linear interpolation), which type of dataset is most appropriate to set these hyper-parameters on?
( ) training data
( ) test data
( ) held-out data

## Q5 (1 Point)
Which of the following is a valid motivation for making a Markov assumption in a language model?
( ) Shorter context reduces the number of parameters and makes learning and generalization easier.
( ) Shorter context lets the model capture long-range interactions.
( ) Longer context lets the model capture long-range interactions.
