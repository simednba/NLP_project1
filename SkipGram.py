# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:33:40 2020

@author: simed
"""


from __future__ import division
import argparse
import pandas as pd

# useful stuff
import spacy
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

spacy_nlp = spacy.load("en_core_web_sm")

__authors__ = ['Hammouch Ouassim','El Hajji Mohamed','POKALA Sai Deepesh', 'de BROGLIE Philibert']
__emails__  = ['mohamed.el-hajji@student.ecp.fr','saideepesh.pokala@student-cs.fr','philibert.de-broglie@student-cs.fr']

def text2sentences(sentences):
	# feel free to make a better tokenization/pre-processing
    
    processed_sentences = []
    for sentence in sentences:
        string = sentence.lower()
        spacy_tokens = spacy_nlp(string)
        lem = spacy.lemmatizer
        string_tokens = [token.lemma_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
        processed_sentences.append(string_tokens)
    return processed_sentences

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.w2id = create_w2id_map(sentences) # word to ID mapping
        self.trainset = set(sentences) # set of sentences
        self.vocab = list(set(self.w2id.keys())) # list of valid words
        self.n = nEmbed
        self.n_neg = negativeRate
        self.winsize = winSize
        self.mincount = minCount
        self.w1 = np.random.uniform(-1, 1, (len(self.vocab), self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, len(self.vocab)))
        

	def sample(self, omit):
		"""samples negative words, ommitting those in set omit"""

		#TODO : add unigram model sampling (**3/4)
		n = len(self.vocab)
		neg_sample_idxs = []
		for i in range(self.n_neg):
			neg_sample_idx = np.random.randint(n)
			while neg_sample_idx in omit :
				neg_sample_idx = np.random.randint(n)
			neg_sample_idxs.append(neg_sample_idx)

		return neg_sample_idxs

    def train(self):
        #TODO : cycle throught each epoch
        for counter, sentence in enumerate(self.trainset):
            sentence = filter(lambda word: word in self.vocab, sentence)

            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                winsize = np.random.randint(self.winSize) + 1
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))

                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx: continue
                    negativeIds = self.sample({wIdx, ctxtId})
                    self.trainWord(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1

            if counter % 1000 == 0:
                print ' > training %d of %d' % (counter, len(self.trainset))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.

    def get_one_hot(self, word):
  		one_hot = np.zeros(len(self.vocab))
  		word_index = self.w2id[word]
  		one_hot[word_index] = 1
  		return one_hot


	def trainWord(self, training_data):

		# Cycle through each epoch
		for i in range(self.epochs):
			# Intialise loss to 0
			self.loss = 0
			# Cycle through each training sample
			# w_t = vector for target word, w_c = vectors for context words
			for w_t, w_c in training_data:
				# Forward pass
				# 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
				y_pred, h, u = self.forward_pass(w_t)
				#########################################
				# print("Vector for target word:", w_t)	#
				# print("W1-before backprop", self.w1)	#
				# print("W2-before backprop", self.w2)	#
				#########################################

				# Calculate error
				# 1. For a target word, calculate difference between y_pred and each of the context words
				# 2. Sum up the differences using np.sum to give us the error for this particular target word
				EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
				#########################
				# print("Error", EI)	#
				#########################

				# Backpropagation
				# We use SGD to backpropagate errors - calculate loss on the output layer 
				self.backward(EI, h, w_t)
				#########################################
				#print("W1-after backprop", self.w1)	#
				#print("W2-after backprop", self.w2)	#
				#########################################

				# Calculate loss
				# There are 2 parts to the loss function
				# Part 1: -ve sum of all the output +
				# Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
				# Note: word.index(1) returns the index in the context word vector with value 1
				# Note: u[word.index(1)] returns the value of the output layer before softmax
				self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
				
				#############################################################
				# Break if you want to see weights after first target word 	#
				# break 													#
				#############################################################
			print('Epoch:', i, "Loss:", self.loss)



	def forward(self, x):
		embedding = np.dot(x, self.w1)
		pred = np.dot(embedding, self.w2)
		proba = self.softmax(pred)
		return proba, embedding, pred
    
    def softmax(self, x):
        interm = np.exp(x-max(x))
        return interm / interm.sum(axis=0)
    

	def backward(self, error, proba, x):
        g = ((proba - y) / x.shape[0]).astype(np.float16)  # Shape (-1, vocab_size)
        grad_w2 = self.h.T.dot(g).astype(np.float16)  # Shape (embed_dim, vocab_size)

        g = g.dot(self.w2.T).astype(np.float16)  # Shape (-1, embed_dim)
        grad_w1 = x.T.dot(g).astype(np.float16)  # Shape (vocab_size, embed_dim)

        grad_w1 = np.clip(grad_w1, 0.01, 20).astype(np.float16)
        grad_w2 = np.clip(grad_w2, 0.01, 20).astype(np.float16)
        
        
        grad_1 = np.array([[i*j for j in error] for i in out1])
        grad_2 = np.array([[i*j for j in np.dot(self.w2,error.transpose())] for i in x])
		self.w1 = self.w1 - (self.lr * grad_2)
		self.w2 = self.w2 - (self.lr * grad_1)
        

	def save(self,path):
		raise NotImplementedError('implement it!')

	def similarity(self,word1,word2):
		"""
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
		word1_emb = self.w2id[word1]
		word2_emb = self.w2id[word2]

		return np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb) * np.linalg.norm(word2_emb))

	@staticmethod
	def load(path):
		raise NotImplementedError('implement it!')
        
def create_w2id_map(sentences):
	w2id = {}
	id = 0
	for sentence in sentences :
		for token in sentence :
			if token not in w2id.keys():
				w2id[token]=id
				id +=1
	return w2id

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()

	if not opts.test:
		sentences = text2sentences(opts.text)
		sg = SkipGram(sentences)
		sg.train(...)
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
		for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(sg.similarity(a,b))
