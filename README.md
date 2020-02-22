# Assignment 1 - SkipGram Implementation with Negative Sampling

### Authors: Hammouch Ouassim, El Hajji Mohamed, POKALA Sai Deepesh, de BROGLIE Philibert

This exercise, is an attempt to generate word embeddings by implementing the skipGram model with Negative-Sampling. SkipGram is one of the two algorithms cited in the Word2vec paper, used to learn such word embeddings. SkipGram, essentially, is a fully connected neural-network that takes in a word as input and predicts the most probable context words for that input word. 


After going through a few papers and trying out various codes, here is how we implemented the skipGram model.

## Default parameters
Here as the default paramters you will find when launching the training:
- Window Size= 5
- Embeddings size = 100
- Number of Negative Samples = 5
- Learning rate = 5e-2

## Task pipeline

Text preprocessing:

- Read all the lines in the given text file.
- Remove all special characters and numbers except spaces and apostrophes. We then convert the apostrophes to spaces as that is the standard practice in NLP tasks (explained in the Deep Learning course by Dr. Vincent Lepetit).
- We then convert all the words in a sentence to lower case and spilt the sentence into a list of words so as to get a list of lists for each text file.
- From this data (which is the trainset), we build our vocabulary which is a list of unique words. 
- The position of each word in this vocab list is the ID of the word that will be used as a pointer to the word in all the matrices that follow. 

Training:

- Obtain positive examples from the neigboring context of a target word, the number of positive examples depends on the window size. 
- Obtain negative examples by randomly sampling words in the lexicon based on a unigram distribution (which is built using words frequency)
- Train the model by optimizing the loss function
- Use the regression weights as the embedding vectors

First of all, a target word will be chosen with its context words, then negative words will be randomly sampled and finally all of theses words will then be train to our model. This means that the model is composed of two weights matrices which contains word embeddings, ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BW%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%20%5Ctimes%20V%7D%20%5Ctext%20%7B%20and%20%7D%20%5Cmathbf%7BC%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BV%20%5Ctimes%20d%7D) with d the embedding dimension and V the number of words. Matrix W which will be updated by the target word only, contains the embeddings that will then be used to compute cosine similarity and for word2vec for example. The C matrix will be used to train context and negative words but its embeddings won’t ever be used after training is done.

The loss is defined as:

![equation](https://latex.codecogs.com/gif.latex?%24%24L%28%5Cboldsymbol%7B%5Ctheta%7D%29%3D%5Csum_%7B%28t%2C%20p%29%20%5Cin&plus;%7D-%5Clog%20%5Cfrac%7B1%7D%7B1&plus;%5Cexp%20%5Cleft%28-%5Cmathbf%7Bw%7D_%7Bt%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bc%7D_%7Bp%7D%5Cright%29%7D&plus;%5Csum_%7B%28t%2C%20n%29%20%5Cin-%7D-%5Clog%20%5Cfrac%7B1%7D%7B1&plus;%5Cexp%20%5Cleft%28%5Cmathbf%7Bw%7D_%7Bt%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bc%7D_%7Bn%7D%5Cright%29%7D%24%24)

where ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D_%7Bt%7D%2C%20%5Cmathbf%7Bc%7D_%7Bt%7D) are the ![equation](https://latex.codecogs.com/gif.latex?%24t%5E%7B%5Ctext%20%7Bth%20%7D%7D%24) column of W and the ![equation](https://latex.codecogs.com/gif.latex?%24t%5E%7B%5Ctext%20%7Bth%20%7D%7D%24) row of C. 

t: target word
p: positive (context) word
n: negative word

The positive words (context words) are denoted by the ‘+’, and the negatives ones by the ‘-’. The higher the result of the dot product between two words vectors, the more similar they are to each other. Hence, the target word and i is, the more similar or closer the two vectors are together. The use of the logistic function transforms this dot product into a probability. Hence for positive words the goal is to maximise this 1/1+exp(−wtTcp) so having a large dot product between wtTcpand this is done by minimising the negative log of this . 

Additionally the goal is to minimise the similarity of target and negative words hence to minimise wtTcn hence maximising 1/1+exp(wtTcn)and therefore minimising the negative log of this. This is why this loss is adapted to the skip-gram model using negative sampling as we try to reduce the loss L by varying the matrix W and C coressponding to the embeddings. 

To do so, stochastic gradient descent is then implemented. 
The loss partial derivatives can be calculated as follows:

![equation](https://latex.codecogs.com/gif.latex?s_%7Bp%7D%3D%5Cfrac%7B1%7D%7B1&plus;%5Cexp%20%5Cleft%28%5Cmathbf%7Bw%7D_%7Bt%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bc%7D_%7Bp%7D%5Cright%29%7D)

dL/dwt = -1/1+exp(wtTcp) * cp + ∑(n) 1/1+exp(-wtTcn) * cn

dL/dcp = -1/1+exp(wtTcp) * wt

dL/dcn = 1/1+exp(-wtTcn) * wt

With those derivates the SGD can be applied as follow with stepsize  α :

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%5Cmathbf%7Bw%7D_%7Bt%7D%20%5Cleftarrow%20%5Cmathbf%7Bw%7D_%7Bt%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bw%7D_%7Bt%7D%7D%5C%5C%20%26%5Cmathbf%7Bc%7D_%7Bp%7D%20%5Cleftarrow%20%5Cmathbf%7Bc%7D_%7Bp%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bc%7D_%7Bp%7D%7D%5C%5C%20%26%5Cmathbf%7Bc%7D_%7Bn%7D%20%5Cleftarrow%20%5Cmathbf%7Bc%7D_%7Bn%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%20L_%7B%28t%2C%20p%29%7D%7D%7B%5Cpartial%20%5Cmathbf%7Bc%7D_%7Bn%7D%7D%20%5Cend%7Baligned%7D)

During the training of each target word, for every context word, our skipGram model samples negatives words and gets the derivatives shown above to compute gradient descent. From there, it changes the weigth matrices W and C and then compute the loss.
 
 
 
