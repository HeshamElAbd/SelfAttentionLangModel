#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:46:46 2019

@author: Hesham El Abd
@Description: Building a simple language model 
"""
import tensorflow as tf
from buildingTransFormer import Modeler
import numpy as np
import pickle
from utility_functions import( EncodeText, PrepareTrainingTensors,loss)
import time 

raw_text=""
with open("../Data/raw.txt","r") as openfile:
    for line in openfile: 
        raw_text+=" \n "+line
raw_text=raw_text.strip()

## mapping: 
text_vocab=sorted(set(raw_text))
char2idx={char:idx for idx, char in enumerate(text_vocab)}
idx2char= np.array(text_vocab)

## save the maps for subsequent uses : 
with open("../Resources/char2idx.pickle","wb") as outputfile:
    pickle.dump(char2idx,outputfile)

with open("../Resources/idx2char.pickle","wb") as outputfile:
    pickle.dump(idx2char,outputfile)

## Encoding the text numerically: 
text_encoded=EncodeText(text=raw_text, encoding_scheme=char2idx)

## prepare the data as a TF DataSet
text_as_chuncks=tf.data.Dataset.from_tensor_slices(text_encoded).batch(
        120+1,drop_remainder=True)
## preapre the TF data for training 
train_dataset=text_as_chuncks.map(PrepareTrainingTensors).shuffle(
        10000).batch(32,drop_remainder=True)

## define the models:
Trial_one=Modeler(embedding_dim=8,vocabulary_size=10000, 
                      conditional_string_length=120,dff=16,num_encoder_layer=2,
                      num_heads=4,rate=0.1)
## loss function: 

## define the optimizer: 
optm=tf.keras.optimizers.Adam()
## define the training objective: 
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


for (batch, (inp, tar)) in enumerate(train_dataset):
    with tf.GradientTape() as tape:
        predictions = Trial_one(inp, 
                                 True, 
                                 None)
        loss_ = loss(tar, predictions)
    gradients = tape.gradient(loss_, Trial_one.trainable_variables)    
    optm.apply_gradients(zip(gradients, Trial_one.trainable_variables))
    print("loss is : " +str(np.mean(loss_.numpy())))




























