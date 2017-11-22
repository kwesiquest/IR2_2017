# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:48:16 2017

@author: Robin Bakker
"""

import tensorflow as tf

class LSE(object):

    def __init__(self,batch_size, word_emb_size, entity_emb_size,vocab_size, vocabulary):
        
        self.word_emb_size = word_emb_size
        self.entity_emb_size = entity_emb_size
        self.vocab_size = vocab_size
        self.vocabulary = vocabulary
        
        self.Wv = tf.get_variable('Wv', shape=(word_emb_size,vocab_size))
        self.W = tf.get_varaible('W',shape=(entity_emb_size,word_emb_size))
        self.b = tf.get_variable('b',shape=(entity_emb_size,1))
        
        
    def getOneHot(self, _, word):
        return tf.one_hot(self.vocabulary[word],self.vocab_size)
        
    def getOneHotAverage(self,words):
        
        one_hots = tf.scan(self.getOneHot,words)
        return tf.reduce_mean(one_hots)

    def getWordEmbedding(self,one_hot_avg):
    
        return tf.matmul(self.Wv,one_hot_avg)
    
    def getEntityEmbedding(self,wordEmbedding):

        return tf.tanh(tf.matmul(self.W,wordEmbedding) + self.b)
    
    def getSimilarity(self,entities, mapping):
        # entities: examples x ev 
        # mapping: 1 x ev
        # res: examples x 1
        return tf.sigmoid(tf.matmul(entities,tf.transpose(mapping)))
    
    def getProbability(self,similar,dissimilar):
        
        return tf.log(similar) + tf.reduce_sum(tf.log(1 - dissimilar))
    
    def loss(self, probability):
        
        return - 1 / self.batch_size # add regularization