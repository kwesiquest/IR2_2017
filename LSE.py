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
        self.e = 1e-5
        
        self.Wv = tf.get_variable('Wv', shape=(vocab_size,word_emb_size))
        self.W = tf.get_variable('W',shape=(word_emb_size,entity_emb_size))
        self.b = tf.get_variable('b',shape=(1,entity_emb_size))
    
    # embedding from W_e
    # input is a batch x entities x docs x ngrams x n tensor
    # returns a batch x entities x e_emb tensor
    def entityEmbedding(self, D):
    
        D = tf.one_hot(self.vocabulary.lookup(D),self.vocab_size)
        D = tf.squeeze(D,axis=4) # b x e x d x n x V
        s = tf.reduce_sum(D,axis=2) # b x e x n x V
        s = tf.cast(tf.reduce_sum(tf.where(s == 0,0,1)),tf.float32)
        D = tf.reduce_sum(D,axis=2)
        D = tf.divide(D,s) # b x e x n x V
        D = tf.tensordot(D, self.Wv,1) # b x e x n x emb
        D = tf.tanh(tf.tensordot(D,self.W,1) + self.b)
        return tf.reduce_sum(D,axis = 2)
    
    # projects ngrams into entity embedding space
    # batch x ngrams x n
    # returns batch x ngrams x emb
    def project(self,ngrams):
        D = tf.one_hot(self.vocabulary.lookup(ngrams),self.vocab_size)
        s = tf.reduce_sum(D,axis=3)
#        s = tf.cast(tf.reduce_sum(tf.where(s == 0,0,1)),tf.float32)
        w = tf.zeros(tf.shape(s))
        s = tf.reduce_sum(tf.cast(tf.equal(w,s),tf.float32))
        D = tf.reduce_sum(D,axis=2) # b x ngrams x V
        D = tf.divide(D,s)
        D = tf.tensordot(D, self.Wv,1)
        D = tf.tanh(tf.tensordot(D,self.W,1) + self.b)
        return D
    
    # returns similarity score 
    # takes a batch x entities x emb similar
    # batch x entities x emb dissimilars
    # batch x ngrams x emb projection
    def similarity(self,similar, dissimilar, projection):
               
        projection = tf.transpose(projection, [0,2,1])
        
        S = tf.sigmoid(tf.matmul(similar, projection))  # batch x entities x ngrams
#        S = tf.expand_dims(S,3) # batch x entities x ngrams
        S = tf.log(S)
        S = tf.reduce_sum(S,1) # batch x ngrams
#        
        SD = tf.sigmoid(tf.matmul(dissimilar,projection)) # batch x entities x ngrams
#        SD = tf.expand_dims(SD,3) # batch x entities x ngrams
#        SD = tf.reduce_sum(SD, axis = 3, keep_dims = True)
        SD = tf.log((1- SD + self.e))
        SD = tf.reduce_sum(SD,1) # batch x ngrams
#        
        return S + SD
#        return SD

    # returns mean loss of batch x ngrams
    def loss(self, similarity):
        return -tf.reduce_mean(similarity)
    
    
    def train_step(self,loss):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)