# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:59:14 2017

@author: Robin Bakker
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import nGramParser as gram

from LSE import LSE as LSE

BATCH_SIZE = None
W_SIZE = 100
E_SIZE = 150
LEARNING_RATE = 1e-3
PATH_TO_DATA = 'D:\\UvA\\IR2\\'
TRAIN_STEPS = 100
N_GRAM_SIZE = 1
DISSIMILAR_AMOUNT = 10

vocabulary = ['a','b','c','d']
vocab_size = len(vocabulary)

print('Loading data')
print('Preprocessing')
print('Finished vocabulary')

ngrams = [['a'],['b'],['-']]
docs = np.array([[['a'],['b'],['-']],[['c'],['d'],['a']]])
dissimilars = [[[['a'],['b'],['-']],[['c'],['d'],['a']]]]
entity_amount = 1

model = LSE(BATCH_SIZE,W_SIZE,E_SIZE,vocab_size,vocabulary,entity_amount, LEARNING_RATE)

ngrams_placeholder = tf.placeholder(tf.string, shape=(None,N_GRAM_SIZE))
documents_placeholder = tf.placeholder(tf.string, shape=(None,None,N_GRAM_SIZE))
dissimilar_placeholder = tf.placeholder(tf.float32, shape=(E_SIZE, None))

entity = model.entityEmbedding(documents_placeholder)
projection = model.project(ngrams_placeholder)
similarity = model.similarity(entity, dissimilar_placeholder, projection)
loss = model.loss(similarity)
train_step = model.train_step(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    tf.tables_initializer().run()
    
    for i in range(TRAIN_STEPS):
        print(i)
    
    #    ngrams, similar, dissimilar = data.getNgrams(path,N_GRAM_SIZE)
    
        ngrams_feed = {ngrams_placeholder: ngrams}
        ngrams_emb = sess.run(projection,feed_dict = ngrams_feed) 
    
        similar = docs
        similar_feed = {documents_placeholder: similar}
        similar_emb = sess.run(entity,feed_dict = similar_feed)
        
        diss = None
        for dissimilar in dissimilars:
            dissimilar_feed = {documents_placeholder: dissimilar}
            dissimilar_emb = sess.run(entity,feed_dict = dissimilar_feed)
            dissimilar_emb = np.expand_dims(dissimilar_emb,1)
            if diss == None:
                diss = dissimilar_emb
            else:
                diss = np.stack(diss,dissimilar_emb, 1)
                
        print(sess.run(loss, feed_dict={ngrams_placeholder: ngrams, documents_placeholder: docs, dissimilar_placeholder: diss}))
        
        sess.run(train_step, feed_dict={ngrams_placeholder: ngrams, documents_placeholder: docs, dissimilar_placeholder: diss})
        
    
    