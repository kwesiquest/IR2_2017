# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:48:16 2017

@author: Robin Bakker
"""

import tensorflow as tf

class LSE(object):

    def __init__(self,batch_size, word_emb_size, entity_emb_size,vocab_size, vocabulary, entity_amount, learning_rate,dissimilar_amount):
        
        self.word_emb_size = word_emb_size
        self.entity_emb_size = entity_emb_size
        self.vocab_size = vocab_size
        self.entity_amount = entity_amount
        self.vocabulary = tf.contrib.lookup.index_table_from_tensor(tf.constant(vocabulary),default_value=1)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.e = 1e-5
        self.d = dissimilar_amount
        
        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)
        
        self.Wv = tf.get_variable('Wv', shape=(batch_size,vocab_size,word_emb_size),initializer=initializer_weights)
        self.W = tf.get_variable('W',shape=(batch_size,word_emb_size,entity_emb_size), initializer=initializer_weights)
        self.b = tf.get_variable('b',shape=(batch_size,1,entity_emb_size), initializer=initializer_biases)
    
    # embedding from W_e
    # input is a b x d x n x w tensor
    def entityEmbedding(self, D):
        
        D = self.vocabulary.lookup(D) # b x d x n x w
        t = tf.cast(tf.equal(D,tf.constant(0,dtype=tf.int64)),tf.float32)
        D = tf.cast(tf.cast(D,tf.float32)-t,tf.int64)
        D = tf.one_hot(D,self.vocab_size) # b x d x n x w x V
        s = tf.reduce_sum(D,axis=4) # b x d x n x w
        s = tf.reduce_sum(s,axis=3,keep_dims=True) + self.e # b x d x n
        
#        s = tf.reduce_sum(D,axis=4) # b x d x n x w
#        w = tf.ones(tf.shape(s))
#        s = tf.reduce_sum(tf.cast(tf.equal(s,w),tf.float32),axis=3) # b x d x n
#        s = tf.expand_dims(s,3) # b x d x n x 1
#        s = tf.reduce_sum(s,1) # b x n x 1
        D = tf.reduce_sum(D,axis=3) # b x d x n x V
#        D = tf.reduce_sum(D,axis=1) # b x n x V
        D = tf.divide(D,s) # b x d x n x V
        D = tf.reduce_mean(D,axis=2) # b x d x V
        D = tf.matmul(D, self.Wv) # b x d x w_emb
        D = tf.tanh(tf.matmul(D,self.W) + self.b) # b x d x e_emb

        return tf.reduce_sum(D,axis=1) #b x e_emb 
    
    # projects ngrams into entity embedding space
    # input is a b x w tensor
    def project(self,ngrams):
        D = self.vocabulary.lookup(ngrams) #b x w
        t = tf.cast(tf.equal(D,tf.constant(0,dtype=tf.int64)),tf.float32)
        D = tf.cast(tf.cast(D,tf.float32)-t,tf.int64)
        D = tf.one_hot(D,self.vocab_size) # b x w x V
        s = tf.reduce_sum(D,axis=2) # b x w
        s = tf.reduce_sum(s,axis=1, keep_dims=True) # b
#        s = tf.reduce_sum(tf.cast(tf.equal(s,tf.constant(1),dtype=tf.float32),tf.float32),axis=1) # b

#        s = tf.expand_dims(s,1) # b x 1
#        s = tf.cast(tf.reduce_sum(tf.where(s == 0,0,1)),tf.float32)
        D = tf.reduce_sum(D,axis=1) # b x V
        D = tf.divide(D,s)
        D = tf.expand_dims(D,1) # b x 1 x V
        D = tf.matmul(D, self.Wv) # b x 1 x w_emb
        D = tf.tanh(tf.matmul(D,self.W) + self.b) # b x 1 x e_emb
        return D
    
    # returns similarity score 
    # input b x e_emb similar
    # input b x e x e_emb dissimilar
    # input b x 1 x e_emb projection
    def similarity(self,similar, dissimilar, projection):
        
        projection = tf.transpose(projection,[0,2,1]) # b x e_emb x 1
#        
        similar = tf.expand_dims(similar,1) # b x 1 x e_emb        
        S = tf.sigmoid(tf.matmul(similar,projection)) #/ (tf.norm(similar)**2 * tf.norm(projection)**2) # b x 1
#        S = tf.log(S + self.e)
        SD = tf.sigmoid(tf.matmul(dissimilar, projection))# / (tf.norm(dissimilar)**2 * tf.norm(projection)**2) # b x e 
#        SD = tf.log((SD + self.e))
#        SD = tf.reduce_sum(SD, axis = 1, keep_dims = True) # b x 1
#        
        logits = tf.squeeze(tf.stack((S,SD),axis=1))
        labels = tf.one_hot([0]*self.batch_size, self.d+1)
        return tf.losses.hinge_loss(labels,logits)

#        return S - SD - 1 - self.d

        
    
    # # returns mean loss of b x 1
    # def loss(self, similarity):
    #     return tf.negative(tf.reduce_mean(similarity))

# returns mean loss of b x 1
    def loss(self, similarity):
        regularizer = tf.contrib.layers.l2_regularizer(0.05)
        reg = tf.contrib.layers.apply_regularization(regularizer,[self.Wv, self.W])
#        return - tf.reduce_mean(similarity) #+ reg
        return similarity # + reg
    
    def train_step(self,loss):
        
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)