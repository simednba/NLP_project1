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
import os
import pickle as pkl
import spacy
import numpy as np
from stopwords import stop_words

__authors__ = ['Hammouch Ouassim','El Hajji Mohamed','POKALA Sai Deepesh', 'de BROGLIE Philibert']
__emails__  = ['ouassim.hammouch@student.ecp.fr', 'mohamed.el-hajji@student.ecp.fr','saideepesh.pokala@student-cs.fr','philibert.de-broglie@student-cs.fr']

def text2sentences(path):
    """
        extract the sentences in the path and preprocess them
        Parameters
        ----------
        path : Path of sentences, str

        Returns
        -------
        processed_sentences : List
            A list with preprocessed sentences split in tokens
    """
    #spacy_nlp = spacy.load("en_core_web_sm")

    with open(path, encoding="utf8") as f:
        sentences = f.readlines()
    sentences = [sentence.rstrip("\n") for sentence in sentences]

    processed_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^a-zA-Z_\s\']+', '', sentence) # remove special characters, numbers
        sentence = [s for s in sentence.lower().replace("'", " ").split() if s not in stop_words]
        processed_sentences.append(sentence) #all lowercase, replace ' with a space and split at each word

    return processed_sentences

def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

def create_w2id_map(sentences):
    """
       Create a word to id mapping
       Parameters
       ----------
       sentences : List
              List of preprocessed sentences
       Returns
       -------
       w2id : Dict
           word to id mapping
    """
    w2id = {}
    id = 0
    for sentence in sentences :
        for token in sentence :
            if token not in w2id.keys():
                w2id[token]=id
                id +=1
    return w2id

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
        self.trainWords = 1
        self.norms = {"W":[], "C":[]}
        self.clip_value = clip_value
        self.display_rate = display_rate

    def sample(self, omit):
        """
        samples negative words, ommitting those in set omit
        :param omit: set of words
        :return: ids of negative samples
        """

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
        """
        Performs the forward pass of a neural network, not used
        Parameters
        ----------
        x : np.array
            data to perform the forward pass
        Returns
        -------
        proba : array
            The probabilities after the softmax layer
        embedding : array
            The hidden layer ( word embeddings)
        pred : array
            The output of the network before the sofrtmax layer
        """
        embedding = np.dot(x, self.w1)
        pred = np.dot(embedding, self.w2)
        proba = self.softmax(pred)
        return proba, embedding, pred

    def train(self):
        """
        	    Train the skipgram model
        	    Returns
        	    -------
        	    None.
        """
        for counter, sentence in enumerate(self.trainset):
            self.trainWords += 1
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

            # if counter % self.display_rate == 0:
                # print (' > training %d of %d' % (counter, len(self.trainset)))
                # self.norms["W"].append(np.linalg.norm(self.W))
                # self.norms["C"].append(np.linalg.norm(self.C))

                # print(f"Loss : {self.accLoss / self.trainWords}")
                # self.trainWords = 0
                # self.accLoss = 0.

    def get_one_hot_from_string(self, word):
        """
        Get the one hot vector corresponding to word
        Parameters
        ----------
        word : str
            a word of the vocabulary
        Returns
        -------
        one_hot : array
            One hot vector of the word
        """
        one_hot = np.zeros(len(self.vocab))
        word_index = self.w2id[word]
        one_hot[word_index] = 1
        return one_hot

    def trainWord(self, target_id, context_id, neg_ids):
        """
        Train the model on a batch of  words
        Parameters
        ----------
        target_id : int
            id of the target word
        context_id : int
            id of the context word
        neg_ids : int
            ids of the negative words
        Returns
        -------
        None.
        """

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

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def softmax(self, x):
        interm = np.exp(x-max(x))
        return interm / interm.sum(axis=0)

    def save(self, path):
        """
        Saves the model on the disk as a dictionary
        Parameters
        ----------
        path : str
            The path to the file which will contain the data.
        Returns
        -------
        None.
        """
        with open(path, "wb") as f:
            pkl.dump(self.__dict__, f)
        # print(f"Saved model to {path} successfully")

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """

        try:
            id_1, id_2 = self.w2id[word1], self.w2id[word2]
            emb1 = self.W[:, id_1]
            emb2 = self.W[:, id_2]
        except KeyError:
            return 0

        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def load(self, path):
        """
        Loads the saved dictionnary onto the model
        Parameters
        ----------
        path : str
            The path for the file.
        Returns
        -------
        None.
        """
        with open(path, "rb") as f:
            self.__dict__ = pkl.load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences, lr=0.0005, display_rate=50)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram([])
        sg.load(opts.model)
        for a, b, _ in pairs:
            print(sg.similarity(a, b))



