# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:33:40 2020

@author: simed
"""

#%%
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
sentences = sentences*150000

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

def create_w2id_map(sentences):
	w2id = {}
	id = 0
	for sentence in sentences :
		for token in sentence :
			if token not in w2id.keys():
				w2id[token]=id
				id +=1
	return w2id

#%%

self = SkipGram()
class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5, lr = 0.05, clip_value = 10, display_rate = 100):
		self.w2id = create_w2id_map(sentences) # word to ID mapping
		self.trainset = sentences # set of sentences
		self.vocab = list(set(self.w2id.keys())) # list of valid words
		self.n = nEmbed
		self.n_neg = negativeRate
		self.winSize = winSize
		self.mincount = minCount
		self.C = np.random.uniform(-1, 1, (len(self.vocab), self.n))
		self.W = np.random.uniform(-1, 1, (self.n, len(self.vocab)))
		self.loss = []
		self.accLoss = 0
		self.lr = lr
		self.trainWords = 0
		self.norms = {"W":[], "C":[]}
		self.clip_value = clip_value
		self.display_rate = display_rate

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
					# print(f"Trained ({word},{context_word})")
					self.trainWords += 1

			if counter % self.display_rate == 0:
				print (' > training %d of %d' % (counter, len(self.trainset)))
				# self.loss.append(self.accLoss / self.trainWords)
				self.norms["W"].append(np.linalg.norm(self.W))
				self.norms["C"].append(np.linalg.norm(self.C))

				# man = sg.W[:, sg.w2id["man"]]
				# woman = sg.W[:, sg.w2id["woman"]]
				# king = sg.W[:, sg.w2id["king"]]
				# queen = sg.W[:, sg.w2id["queen"]]
				cosine = 0
				# cosine = np.dot((man - woman), (king - queen)) / (np.linalg.norm(man - woman) * np.linalg.norm(king - queen))

				print(f"Loss : {self.accLoss / self.trainWords} | cosine : {cosine}")
				self.trainWords = 0
				self.accLoss = 0.

				# Save regularly in case the code crashes
				save_path = f"./train_{counter}"
				self.save(save_path)

	def get_one_hot_from_string(self, word):
		one_hot = np.zeros(len(self.vocab))
		word_index = self.w2id[word]
		one_hot[word_index] = 1
		return one_hot


	def trainWord(self, target_id, context_id, neg_ids):
		self.loss = 0
		dw = 0

		w_t = self.W[:,target_id]
		c_p = self.C[context_id,:]
		self.loss -= np.log(self.sigmoid(np.dot(w_t,c_p)))
		# Updates
		dcp = np.clip(self.sigmoid(-1*np.dot(w_t,c_p)), -1*self.clip_value, self.clip_value)
		self.C[context_id, :] = self.C[context_id,:] + self.lr*dcp*w_t
		dw += dcp*c_p

		for neg_id in neg_ids:
			c_n = self.C[neg_id, :]
			self.loss -= np.log(self.sigmoid(-1*np.dot(w_t,c_n)))
		# 	 Updates
			dcn = np.clip(self.sigmoid(np.dot(w_t, c_n)), -1*self.clip_value, self.clip_value)
			self.C[neg_id, :] = self.C[neg_id, :] - self.lr * dcn * w_t
			dw -= dcn*c_n

		self.W[:, target_id] += self.lr*np.clip(dw, -1*self.clip_value, self.clip_value)
		self.accLoss += self.loss

		# print(f"dcp = {dcp} | dw = {np.linalg.norm(dw)}")

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
			id_1, id_2 = self.w2id[word1], self.w2id[word2]
			emb1 = self.W[:, id_1]
			emb2 = self.W[:, id_2]
			#_,word1_emb,_ = self.forward(word1)
			#_,word2_emb,_ = self.forward(word2)
			#word2_emb = final_dict[word2]
		except KeyError:
			return -1

		return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

	@staticmethod
	def load(path):
		raise NotImplementedError('implement it!')


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
		sg = SkipGram(sentences_, lr=0.0005, display_rate=50)
		sg.train()
		sg.save(opts.model)

	else:
		testPath = "C:\\Users\\User\Documents\GitHub\\NLP_project1\EN-SIMLEX-999.txt"
		# pairs = loadPairs(opts.text)
		pairs = loadPairs(testPath)

		sg = SkipGram.load(opts.model)
		for a,b,_ in pairs:
			# make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(sg.similarity(a,b))



