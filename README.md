NLP_project1
SkipGram Implementation with Negative Sampling using numpy

Authors: Hammouch Ouassim, El Hajji Mohamed, POKALA Sai Deepesh, de BROGLIE Philibert

This exercise, is an attempt to create our own word embedding by implementing the skipGram model using numpy library. skipGram is one of the two algorithms cited in the Word2vec paper, used to learn such word embeddings with the CBOW algorithm. skipGram predicts context (or neighbours) words from a centered (or target) one. This mean looking at words around a target one according to a window size, to understand their context, concept etc… 
Basically, skipGram takes as input the centered word’s one hot encoded vector passes it through the hidden layer made of the weight matrix (aka the embeddings). Then the same happens to context words and are then inputted to the softmax classifier. The softmax then gives a probability of being a context word or not. 
In order to reduce computation time negative sampling was implemented and turned out to also increase accuracy, hence will be done here as well. The concept behind negative sampling is to randomly sample words(5-20) from the available vocabulary, which are different from target and context words, and use them for training as examples which should not be outputted with a high probability. Hence context words will be given a probability after softmax of 1 and negative samples of 0.

After going through a few papers and trying out various codes, it was decided to implement a slightly different method to skipgram which will be explained here.

First of all, a target word will be chosen with its context words, then negative words will be randomly sampled and finally all of theses words will then be train to our model. This means that the model is composed of two weights matrices which contains word embeddings, WdVandCVd with d the embedding dimension and V the number of words. Matrix W which will be updated by the target word only, contains the embeddings that will then be used to compute cosine similarity and for word2vec for example. The C matrix will be used to train context and negative words but its embeddings won’t ever be used after training is done.
The following loss will be used:

L(θ)=∑(t,p)∈+−log(1/1+exp(−wtTcp))+∑(t,n)∈-−log(1/1+exp(wtTcn))
t: target word
p: positive (context) word
n: negative word

The positive words (context words) are denoted by the ‘+’, and the negatives ones by the ‘-’. The higher the result of the dot product between two words vectors, the more similar they are to each other. Hence, the target word and i is, the more similar or closer the two vectors are together. The use of the logistic function transforms this dot product into a probability. Hence for positive words the goal is to maximise this 1/1+exp(−wtTcp)so having a large dot product between wtTcpand this is done by minimising the negative log of this . Additionally the goal is to minimise the similarity of target and negative words hence to minimise wtTcn hence maximising 1/1+exp(wtTcn)and therefore minimising the negative log of this. This is why this loss is adapted to the skip-gram model using negative sampling as we try to reduce the loss L by varying the matrix W and C coressponding to the embeddings. 

To do so, stochastic gradient descent is then implemented. 
 
 
 
