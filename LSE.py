# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:48:16 2017

@author: Robin Bakker
"""

import tensorflow as tf

class LSE(object):

    def __init__(self,batch_size, word_emb_size, entity_emb_size,vocab_size, vocabulary, entity_amount, learning_rate):
        
        self.word_emb_size = word_emb_size
        self.entity_emb_size = entity_emb_size
        self.vocab_size = vocab_size
        self.entity_amount = entity_amount
        self.vocabulary = tf.contrib.lookup.index_table_from_tensor(tf.constant(vocabulary),num_oov_buckets=1, default_value=-1)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.Wv = tf.get_variable('Wv', shape=(vocab_size,word_emb_size))
        self.W = tf.get_variable('W',shape=(word_emb_size,entity_emb_size))
        self.b = tf.get_variable('b',shape=(1,entity_emb_size))
    
    # embedding from W_e
    # input is a docs x ngrams x n tensor
    def entityEmbedding(self, D):
        D = tf.one_hot(self.vocabulary.lookup(D),self.vocab_size)
        D = tf.squeeze(D,axis=2)
        s = tf.reduce_sum(D,axis=0)
        s = tf.cast(tf.reduce_sum(tf.where(s == 0,0,1)),tf.float32)
        D = tf.reduce_sum(D,axis=1)
        D = tf.divide(D,s)
        D = tf.matmul(D, self.Wv)
        D = tf.tanh(tf.matmul(D,self.W) + self.b)
        return tf.reduce_sum(D,axis = 0)
    
    # projects ngrams into entity embedding space
    # ngrams x n
    def project(self,ngrams):
        D = tf.one_hot(self.vocabulary.lookup(ngrams),self.vocab_size)
        D = tf.squeeze(D,axis=1)
        s = tf.reduce_sum(D,axis=0)
        s = tf.cast(tf.reduce_sum(tf.where(s == 0,0,1)),tf.float32)
        D = tf.divide(D,s)
        D = tf.matmul(D, self.Wv)
        D = tf.tanh(tf.matmul(D,self.W) + self.b)
        return D
    
    # returns similarity score 
    # takes a e_emb similar, e_emb dissimilar x docs, ngrams x e_emb projection
    def similarity(self,similar, dissimilar, projection):
        
        similar = tf.expand_dims(similar,1)
        S = tf.sigmoid(tf.matmul(projection, similar)) # ngrams x 1
        S = tf.log(S)
        
        SD = tf.sigmoid(tf.matmul(projection, dissimilar)) # ngrams x docs
        SD = tf.reduce_sum(SD, axis = 1, keep_dims = True) # ngrams x 1
        SD = tf.log((1- SD))
        
        return S + SD
    
    def getProbability(self,similar,dissimilar):
        
        return tf.log(similar) + tf.reduce_sum(tf.log(1 - dissimilar))
    
    # returns mean loss of ngrams x 1
    def loss(self, similarity):
        return - tf.reduce_mean(similarity)
    
    def train_step(self,loss):
        
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)