# Assignment 1 - SkipGram Implementation with Negative Sampling

### Authors: Hammouch Ouassim, El Hajji Mohamed, POKALA Sai Deepesh, de BROGLIE Philibert

This exercise, is an attempt to generate word embeddings by implementing the skipGram model with Negative-Sampling. SkipGram is one of the two algorithms cited in the Word2vec paper, used to learn such word embeddings. SkipGram, essentially, is a fully connected neural-network with one layer that takes in a word as input and predicts the most probable context words for that input word. 
However, the end use of this network will not be done this way! So, the final goal is to force the model to learn a representation of each word (represented by the value of the neurons of the hidden layer), and then use this representation for other tasks.


After going through a few papers and trying out various codes, here is how we implemented the skipGram model.

## Default parameters
Here as the default paramters you will find when launching the training:
- Window Size= 2
- Embeddings size = 100
- Number of Negative Samples per Positive Sample = 4
- Learning rate = 5e-4

## Task pipeline

Text preprocessing:

- Read all the lines in the given text file.
- Remove all special characters and numbers except spaces and apostrophes. We then convert the apostrophes to spaces as that is the standard practice in NLP tasks (explained in the Deep Learning course by Dr. Vincent Lepetit).
- We then convert all the words in a sentence to lower case and spilt the sentence into a list of words so as to get a list of lists for each text file.
- We then delete the stop words. We chose to use a list of stop words used by the NLTK library. To do this, we simply defined them in a list.
- From this data (which is the trainset), we build our vocabulary which is a list of unique words. 
- The position of each word in this vocab list is the ID of the word that will be used as a pointer to the word in all the matrices that follow. 

For this step, we tried and compared this preprocessing with the one done by spacy, which you will find in the code but commented. We finally preferred to choose Regexp for its flexibility.

Training:

- Obtain positive examples from the neigboring context of a target word, the number of positive examples depends on the window size. 
- Obtain negative examples by randomly sampling words in the lexicon based on a unigram distribution (which is built using words frequency)
- Train the model by optimizing the loss function


First of all, a target word will be chosen with its context words, then negative words will be randomly sampled and finally all of theses words will then be fed to our model. This means that the model is composed of two weights matrices which contains word embeddings, ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BW%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%20%5Ctimes%20V%7D%20%5Ctext%20%7B%20and%20%7D%20%5Cmathbf%7BC%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BV%20%5Ctimes%20d%7D) with d the embedding dimension and V the number of words. Matrix W which will be updated by the target word only, contains the embeddings that will then be used to compute cosine similarity for word2vec for example. The C matrix will be used to train context and negative words but its embeddings won’t ever be used after training is done.



For the training, the loss is defined as:

![equation](https://latex.codecogs.com/gif.latex?%24%24L%28%5Cboldsymbol%7B%5Ctheta%7D%29%3D%5Csum_%7B%28t%2C%20p%29%20%5Cin&plus;%7D-%5Clog%20%5Cfrac%7B1%7D%7B1&plus;%5Cexp%20%5Cleft%28-%5Cmathbf%7Bw%7D_%7Bt%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bc%7D_%7Bp%7D%5Cright%29%7D&plus;%5Csum_%7B%28t%2C%20n%29%20%5Cin-%7D-%5Clog%20%5Cfrac%7B1%7D%7B1&plus;%5Cexp%20%5Cleft%28%5Cmathbf%7Bw%7D_%7Bt%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bc%7D_%7Bn%7D%5Cright%29%7D%24%24)

where ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D_%7Bt%7D%2C%20%5Cmathbf%7Bc%7D_%7Bt%7D) are the ![equation](https://latex.codecogs.com/gif.latex?%24t%5E%7B%5Ctext%20%7Bth%20%7D%7D%24) column of W and the ![equation](https://latex.codecogs.com/gif.latex?%24t%5E%7B%5Ctext%20%7Bth%20%7D%7D%24) row of C. 

where t, c, p, n are used to denote a target word, a word in the target word's context, a positive sample (word belongs to the context) and a negative sample (word do not belong to the context) respectively.

Here, only one context word is chosen at a time.
The positive words (context words) is denoted by the ‘+’, and the negatives ones by the ‘-’. The higher the result of the dot product between two words vectors, the more similar they are to each other. Hence, the target word and i is, the more similar or closer the two vectors are together. The use of the logistic function transforms this dot product into a probability. Hence for positive words the goal is to maximise this 1/1+exp(−wtTcp) so having a large dot product between wtTcpand this is done by minimising the negative log of this . 

As stated, the purpose of this training is to train the weights between the input layer and the first hidden layer,as the inputs are only one-hot vectors, multiplying the input with the weight matrix only selects a column of this weight matrix. Similarly, the expression of the loss involves only one hot context vector multiplication with the second weight matrix, so in our implementation, we did not use the forward function, but simply selected the rows/columns of interest at each step.

The loss partial derivatives can be calculated as follows:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bw%7D_%7Bt%7D%7D%3D-s_%7Bp%7D%20%5Cmathbf%7Bc%7D_%7Bp%7D&plus;%5Csum_%7Bn%20%5Cin%20%5Cmathcal%7BN%7D%28t%2C%20p%29%7D%20s_%7Bn%7D%20%5Cmathbf%7Bc%7D_%7Bn%7D%5C%5C%20%26%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bc%7D_%7Bp%7D%7D%3D-s_%7Bp%7D%20%5Cmathbf%7Bw%7D_%7Bt%7D%5C%5C%20%26%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bc%7D_%7Bn%7D%7D%3Ds_%7Bn%7D%20%5Cmathbf%7Bw%7D_%7Bt%7D%20%5Cquad%20%5Cforall%20n%20%5Cin%20%5Cmathcal%7BN%7D%28t%2C%20p%29%20%5Cend%7Baligned%7D)

where ![equation](https://latex.codecogs.com/gif.latex?s_%7Bp%7D%3D%5Cfrac%7B1%7D%7B1&plus;%5Cexp%20%5Cleft%28%5Cmathbf%7Bw%7D_%7Bt%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bc%7D_%7Bp%7D%5Cright%29%7D) and ![equation](https://latex.codecogs.com/gif.latex?s_%7Bn%7D%3D%5Cfrac%7B1%7D%7B1&plus;%5Cexp%20%5Cleft%28-%5Cmathbf%7Bw%7D_%7Bt%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bc%7D_%7Bn%7D%5Cright%29%7D)

### Optimisation

With those derivates the SGD can be applied as follow with stepsize  α :

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%5Cmathbf%7Bw%7D_%7Bt%7D%20%5Cleftarrow%20%5Cmathbf%7Bw%7D_%7Bt%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bw%7D_%7Bt%7D%7D%5C%5C%20%26%5Cmathbf%7Bc%7D_%7Bp%7D%20%5Cleftarrow%20%5Cmathbf%7Bc%7D_%7Bp%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bc%7D_%7Bp%7D%7D%5C%5C%20%26%5Cmathbf%7Bc%7D_%7Bn%7D%20%5Cleftarrow%20%5Cmathbf%7Bc%7D_%7Bn%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bc%7D_%7Bn%7D%7D%20%5Cend%7Baligned%7D)

During the training of each target word, for every context word, our skipGram model samples negatives words and gets the derivatives shown above to compute gradient descent. From there, it compute the loss and then changes the weigth matrices W and C.
 

## Bibliography

For this exercise, we read and used the following sources: 

-http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
 http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
 https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b
 https://hangle.fr/post/word2vec/
 https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c
