#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:46:46 2019

@author: Hesham El Abd
@Description: Building a simple language model 
"""
import tensorflow as tf
from TestingTheEncoder import Modeler

Trial_one=mod=Modeler(embedding_dim=8,vocabulary_size=10000, 
                      conditional_string_length=120,dff=16,num_encoder_layer=2,
                      num_heads=4,rate=0.1)
optimizer=tf.keras.optimizers.Adam()
lossFunction=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions, _ = Trial_one(inp,
                                 True, None)
        loss = lossFunction(tar, predictions)
    gradients = tape.gradient(loss, Trial_one.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, Trial_one.trainable_variables))
  
    train_loss(loss)
    train_accuracy(tar, predictions)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
