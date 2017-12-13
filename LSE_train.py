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
import copy 
from LSE import LSE as LSE
import operator

   
def pad_dissimilar(dissimilars):
    
    doc_length = 0
    entity_length = 0
    
    for entities in dissimilars:
    
        for entity in entities:
            leng = np.amax([len(doc) for doc in entity])
            if leng > doc_length:
                doc_length = leng
        
        leng = np.amax([len(entity) for entity in entities])
        if leng > entity_length:
            entity_length = leng
    
    for entities in dissimilars:
    
        for entity in entities:
            
            while len(entity) < entity_length:
                entity.append([['z3r0'] * N_GRAM_SIZE])
            
            pad_entity(entity,leng=doc_length)


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

BATCH_SIZE = 2
W_SIZE = 100
E_SIZE = 150
LEARNING_RATE = 1e-3
PATH_TO_DATA = '/home/sdemo210/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
#PATH_TO_DATA = 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
TRAIN_STEPS = 1000
N_GRAM_SIZE = 3
DISSIMILAR_AMOUNT = 5

print('Loading data')
data = edict(PATH_TO_DATA,N_GRAM_SIZE,True)
print('Preprocessing')
vocab = Vocab(PATH_TO_DATA)
#vocabulary = ['z3r0','a','b','c','d']
vocabulary = ['z3r0','Unk']

full_vocab = vocab.vocab_freq
sorted_vocab = sorted(full_vocab.items(), key=operator.itemgetter(1), reverse=True)
full_vocab = [tup[0] for tup in sorted_vocab]
print("full vocab size:", len(full_vocab))
vocabulary += full_vocab[:20000]


#vocabulary += list(vocab.token2idx.keys())[:1000]
vocab_size = len(vocabulary)
print('Finished vocabulary')

print('Vocab size:', vocab_size)

#A = [['a'],['z3r0']]
#B = [[[['a'],['b']],[['c'],['d'],['a']]],[[['a'],['b']],[['c'],['d'],['a']]]]
#C = [[[[['a'],['b']],[['c'],['d'],['a']]]],[[[['a'],['b']],[['c'],['d'],['a']]]]]

entity_amount = 1

print('Initializing Model')
model = LSE(BATCH_SIZE,W_SIZE,E_SIZE,vocab_size,vocabulary,entity_amount, LEARNING_RATE)
print('Finished Model')

ngrams_placeholder = tf.placeholder(tf.string, shape=(None,N_GRAM_SIZE))
documents_placeholder = tf.placeholder(tf.string, shape=(None,None,None,N_GRAM_SIZE))
dissimilar_placeholder = tf.placeholder(tf.float32, shape=(None, None, E_SIZE))

global_step = tf.Variable(0)

entity = model.entityEmbedding(documents_placeholder)
projection = model.project(ngrams_placeholder)
similarity = model.similarity(entity, dissimilar_placeholder, projection)
loss = model.loss(similarity)
train_step = model.train_step(loss)

init = tf.global_variables_initializer()

tf.summary.scalar('Loss', loss)
merged = tf.summary.merge_all()
trainWriter = tf.summary.FileWriter('summaries/train/run11')

print('Start Training')
with tf.Session() as sess:
    sess.run(init)
    tf.tables_initializer().run()
    
    
    for i in range(TRAIN_STEPS):
        print("train step:", i)
        batch = data.get_random_batch(DISSIMILAR_AMOUNT,BATCH_SIZE)

        
        ngrams = copy.deepcopy(batch.docs)
        similar = copy.deepcopy(batch.similars)
        dissimilars = copy.deepcopy(batch.dissimilars)
#        ngrams = copy.deepcopy(A)
#        similar = copy.deepcopy(B)
#        dissimilars = copy.deepcopy(C)
        
        pad_entity(ngrams)             
        pad_entities(similar)
        pad_dissimilar(dissimilars)
        
        
#        print(np.array(ngrams).shape)
#        print(np.array(similar).shape)
#        print(np.array(dissimilars).shape)
    
#        ngrams_feed = {ngrams_placeholder: ngrams}
#        ngrams_emb = sess.run(projection,feed_dict = ngrams_feed) 
#        print(ngrams_emb.shape)
###        
#        break
        
#        print(np.sum(ngrams_emb))
        
###    
#        similar_feed = {documents_placeholder: similar}
#        similar_emb = sess.run(entity,feed_dict = similar_feed)
#        print(similar_emb.shape)
#        break
#        
#        dissimilar_feed = {documents_placeholder: dissimilars}
#        dissimilar_emb = sess.run(entity, feed_dict = dissimilar_feed)
        
        
        dissimilars = np.array(dissimilars)
        dissimilars = np.transpose(dissimilars,[1,0,2,3,4])
#        print(dissimilars.shape)
        
        diss = None
        new = True
        for dissimilar in dissimilars:
            dissimilar_feed = {documents_placeholder: dissimilar}
            dissimilar_emb = sess.run(entity,feed_dict = dissimilar_feed)
            dissimilar_emb = np.expand_dims(dissimilar_emb,1)
            if new == True:
                diss = dissimilar_emb
                new = False
            else:
                diss = np.concatenate((diss,dissimilar_emb), 1)
        #print('----')
#        print(diss.shape)
        
        
        
#        print(sess.run(similarity,feed_dict={ngrams_placeholder: ngrams, documents_placeholder: similar, dissimilar_placeholder: diss})
        #print("loss:", sess.run(loss, feed_dict={ngrams_placeholder: ngrams, documents_placeholder: similar, dissimilar_placeholder: diss}))
        sess.run(train_step, feed_dict={ngrams_placeholder: ngrams, documents_placeholder: similar, dissimilar_placeholder: diss})
        summ, e_loss = sess.run([merged, loss], feed_dict={ngrams_placeholder: ngrams, documents_placeholder: similar, dissimilar_placeholder: diss})
	print("loss:", e_loss)
        trainWriter.add_summary(summ, global_step=i)
    
    
