# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:33:40 2020

@author: simed
"""


from __future__ import division
import argparse
import pandas as pd
import re

# useful stuff
import spacy
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
from stopwords import stop_words

path = "C:\\Users\\User\Documents\GitHub\\NLP_project1\\news.en-00001-of-00100"

__authors__ = ['Hammouch Ouassim','El Hajji Mohamed','POKALA Sai Deepesh', 'de BROGLIE Philibert']
__emails__  = ['mohamed.el-hajji@student.ecp.fr','saideepesh.pokala@student-cs.fr','philibert.de-broglie@student-cs.fr']

sentences = ["Elk calling -- a skill that hunters perfected long ago to lure game with the promise of a little romance -- is now its own sport .Don 't !",
			 "Fish , ranked 98th in the world , fired 22 aces en route to a 6-3 , 6-7 ( 5 / 7 ) , 7-6 ( 7 / 4 ) win over seventh-seeded Argentinian David Nalbandian ."]

def text2sentences(sentences):
	# feel free to make a better tokenization/pre-processing
	spacy_nlp = spacy.load("en_core_web_sm")
	processed_sentences = []
	for sentence in sentences:
		sentence = re.sub(r'[^a-zA-Z_\s\']+', '', sentence) # remove special characters, numbers
		sentence = [s for s in sentence.lower().replace("'", " ").split() if s not in stop_words]
		processed_sentences.append(sentence) #all lowercase, replace ' with a space and split at each word
		#string = sentence.lower()
		#spacy_tokens = spacy_nlp(string)
		#lem = spacy.lemmatizer
		#string_tokens = [token.lemma_ for token in spacy_tokens if not token.is_punct if not token.is_stop if not token.pos_ == 'NUM']
		#processed_sentences.append(string_tokens)
	return processed_sentences

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs


class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
		self.w2id = create_w2id_map(sentences) # word to ID mapping
		self.trainset = sentences # set of sentences
		self.vocab = list(set(self.w2id.keys())) # list of valid words
		self.n = nEmbed
		self.n_neg = negativeRate
		self.winSize = winSize
		self.mincount = minCount
		self.w1 = np.random.uniform(-1, 1, (len(self.vocab), self.n))
		self.w2 = np.random.uniform(-1, 1, (self.n, len(self.vocab)))
		self.loss = []
		self.accLoss = 0
		self.lr = 0.05
		self.trainWords = 0



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

	def forward(self, x):
		embedding = np.dot(x, self.w1)
		pred = np.dot(embedding, self.w2)
		proba = self.softmax(pred)
		return proba, embedding, pred

	def train(self):
		#TODO : cycle throught each epoch
		for counter, sentence in enumerate(self.trainset):
			sentence = list(filter(lambda word: word in self.vocab, sentence))

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

			if counter % 1 == 0:
				print (' > training %d of %d' % (counter, len(self.trainset)))
				# self.loss.append(self.accLoss / self.trainWords)
				print(f"Loss : {self.accLoss / self.trainWords}")
				self.trainWords = 0
				self.accLoss = 0.

	def get_one_hot_from_string(self, word):
		one_hot = np.zeros(len(self.vocab))
		word_index = self.w2id[word]
		one_hot[word_index] = 1
		return one_hot


	def trainWord(self, target_id, context_id, neg_ids):
		self.loss = 0
		w_t = np.array([0]*len(self.vocab))
		w_t[target_id] = 1
		proba, embeddings, predictions = self.forward(w_t)
		self.loss= -np.log(self.sigmoid(np.dot(predictions.T,w_t)))
		for id_ in neg_ids:
			w_c = np.array([0]*len(self.vocab))
			w_c[id_] = 1
			self.loss -=np.log(self.sigmoid(-np.dot(predictions.T,w_c)))
		self.accLoss+= self.loss
		all_ids = [id_ for id_ in neg_ids]
		all_ids.append(context_id)
		w1_grad = 0
		for index,id_ in enumerate(all_ids):
			tj = 0
			if index == len(all_ids)-1:
				tj = 1
			w_c = np.array([0]*len(self.vocab))
			w_c[id_] = 1
		# 	self.w2[id_, :] -= self.lr * (self.sigmoid(np.dot(predictions, w_c)) - tj) * w_c
		# 	w1_grad += (self.sigmoid(np.dot(predictions, w_t)) - tj) * predictions
		# self.w1[:, target_id] -= self.lr * w1_grad
			self.w2[:, id_] -= self.lr*(self.sigmoid(np.dot(predictions, w_c))-tj)*embeddings
			w1_grad += (self.sigmoid(np.dot(predictions,w_t))-tj)*predictions
		self.w1[ :, target_id] -= self.lr*w1_grad



	@staticmethod
	def sigmoid(x):
		return 1/(1+np.exp(-x))



	def softmax(self, x):
		interm = np.exp(x-max(x))
		return interm / interm.sum(axis=0)




	def save(self,path):
		raise NotImplementedError('implement it!')

	def similarity(self,word1,word2):
		"""
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""

		try:
			#word1_emb = final_dict[word1]
			_,word1_emb,_ = self.forward(word1)
			_,word2_emb,_ = self.forward(word2)
			#word2_emb = final_dict[word2]
		except KeyError:
			return -1

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
		with open(path, 'r', encoding="utf8") as f :
			sentences = f.readlines()
		sentences = [sentence.rstrip('\n') for sentence in sentences]
		sentences_ = text2sentences(sentences)
		sg = SkipGram(sentences_)
		sg.train()
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
		for a,b,_ in pairs:
			# make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(sg.similarity(a,b))
