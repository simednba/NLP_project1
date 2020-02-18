







from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['author1','author2','author3']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

path = "C:\\Users\\User\Desktop\Centrale\Scolarité\\4A\Cours\SBA\dataset.txt"
sentences = text2sentences(path)

# For training
def text2sentences(path):
	# feel free to make a better tokenization/pre-processing
	sentences = []
	with open(path) as f:
		for l in f:
			sentences.append( l.lower().split("|") )
	return sentences

# For testing
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

w2id = create_w2id_map(sentences)
vocab = list(w2id.keys())


class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
		self.w2id = create_w2id_map(sentences) # word to ID mapping
		self.trainset = sentences # set of sentences
		self.vocab = list(self.w2id.keys()) # list of valid words
		self.negativeRate = negativeRate

	def sample(self, omit):
		"""samples negative words, ommitting those in set omit"""

		# To do : add unigram model sampling (**3/4)

		n = len(self.vocab)
		neg_sample_idxs = []

		for i in range(self.negativeRate):
			neg_sample_idx = np.random.randint(n)
			while neg_sample_idx in omit :
				neg_sample_idx = np.random.randint(n)
			neg_sample_idxs.append(neg_sample_idx)

		return neg_sample_idxs


	def train(self):
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

	def trainWord(self, wordId, contextId, negativeIds):

		# Faire Forward puis recuprer embeddings pour tous les couples sous u_w, u_c, u_n_list
		loss = np.log(sigmoid(np.dot(u_w, u_c)))
		for u_n in u_n_list :
			loss += np.log(sigmoid(-1*np.dot(u_w, u_n)))


		raise NotImplementedError('here is all the fun!')

	def save(self,path):
		raise NotImplementedError('implement it!')

	def similarity(self,word1,word2):
		"""
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
		raise NotImplementedError('implement it!')

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

skipGram.py
Affichage de skipGram.py

