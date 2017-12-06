# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:59:14 2017

@author: Robin Bakker
"""

import numpy as np
import tensorflow as tf
from nGramParser import EntityDict as edict
from nGramTokenMap import Vocab
import sys

from LSE import LSE as LSE

   
def pad_entities(entities):
    
    doc_length = 0
    
    for entity in entities:
        leng = np.amax([len(doc) for doc in entity])
        if leng > doc_length:
            doc_length = leng
    
    entity_length = np.amax([len(entity) for entity in entities])
    
    for entity in entities:
        
        while len(entity) < entity_length:
            entity.append([['z3r0'] * N_GRAM_SIZE])
        
        pad_entity(entity,leng=doc_length)
        
                
def pad_entity(entity,leng=0):
    if leng == 0: 
        leng = np.amax([len(doc) for doc in entity])
    
    for doc in entity:
        while len(doc) < leng:
            doc.append(['z3r0'] * N_GRAM_SIZE)

BATCH_SIZE = 10
W_SIZE = 100
E_SIZE = 150
LEARNING_RATE = 1e-3
PATH_TO_DATA = 'reviews_Home_and_Kitchen_5.json.gz'
TRAIN_STEPS = 1000
N_GRAM_SIZE = 1
DISSIMILAR_AMOUNT = 1

print('Loading data')
data = edict(PATH_TO_DATA,N_GRAM_SIZE,True)
print('Preprocessing')
vocab = Vocab(PATH_TO_DATA)
vocabulary = list(vocab.token2idx.keys())[:5000]
vocab_size = len(vocabulary)
print('Finished vocabulary')
#vocabulary = ['a','b','c','d']
print('Vocab size:', vocab_size)

#ngrams = [['a'],['b'],['-']]
#docs = np.array([[['a'],['b'],['-']],[['c'],['d'],['a']]])
#dissimilars = [[[['a'],['b'],['-']],[['c'],['d'],['a']]]]
entity_amount = 1

print('Initializing Model')
model = LSE(BATCH_SIZE,W_SIZE,E_SIZE,vocab_size,vocabulary,entity_amount, LEARNING_RATE)
print('Finished Model')

ngrams_placeholder = tf.placeholder(tf.string, shape=(None,N_GRAM_SIZE))
documents_placeholder = tf.placeholder(tf.string, shape=(None,None,N_GRAM_SIZE))
dissimilar_placeholder = tf.placeholder(tf.float32, shape=(E_SIZE, None))

global_step = tf.Variable(0)

entity = model.entityEmbedding(documents_placeholder)
projection = model.project(ngrams_placeholder)
similarity = model.similarity(entity, dissimilar_placeholder, projection)
loss = model.loss(similarity)
train_step = model.train_step(loss)

init = tf.global_variables_initializer()

tf.summary.scalar('Loss', loss)
merged = tf.summary.merge_all()
trainWriter = tf.summary.FileWriter('summaries/train/run1')

print('Start Training')
with tf.Session() as sess:
    sess.run(init)
    tf.tables_initializer().run()
    
    batch = data.get_random_batch(DISSIMILAR_AMOUNT,BATCH_SIZE)
    
    for i in range(TRAIN_STEPS):
        print(i)
        
       
        ngrams = batch.docs
        similar = batch.similars
        dissimilars = batch.dissimilars
        
        pad_entity(ngrams)
        print(np.array(ngrams).shape)
                
        pad_entities(similar)
        print(np.array(similar).shape)
    
        ngrams_feed = {ngrams_placeholder: ngrams}
        ngrams_emb = sess.run(projection,feed_dict = ngrams_feed) 
    
        similar_feed = {documents_placeholder: similar}
        similar_emb = sess.run(entity,feed_dict = similar_feed)

        dissimilar_feed = {documents_placeholder: dissimilars}
        dissimilar_emb = sess.run(entity, feed_dict = dissimilar_feed)
        
        
        
        diss = None
        for dissimilar in dissimilars:
            pad_entities(dissimilar)
            print(np.array(dissimilar).shape)
            sys.exit(0)
            dissimilar = np.array(dissimilar)
#            print('D',dissimilar.shape)
            dissimilar_feed = {documents_placeholder: dissimilar}
            dissimilar_emb = sess.run(entity,feed_dict = dissimilar_feed)
            dissimilar_emb = np.expand_dims(dissimilar_emb,1)
            if diss == None:
                diss = dissimilar_emb
            else:
                diss = np.stack(diss,dissimilar_emb, 1)
        #print('----')
        #print(diss)
        

        
        #print(sess.run(similarity,feed_dict={ngrams_placeholder: ngrams, documents_placeholder: similar, dissimilar_placeholder: diss}))
        print(sess.run(loss, feed_dict={ngrams_placeholder: ngrams, documents_placeholder: similar, dissimilar_placeholder: diss}))
        sess.run(train_step, feed_dict={ngrams_placeholder: ngrams, documents_placeholder: similar, dissimilar_placeholder: diss})
        summ, e_loss = sess.run([merged, loss], feed_dict={ngrams_placeholder: ngrams, documents_placeholder: similar, dissimilar_placeholder: diss})
        trainWriter.add_summary(summ, global_step=i)
    
    