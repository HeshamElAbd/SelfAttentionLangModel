#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 13:23:41 2019

@author: Hesham El Abd
"""
from SelfAttentionLangModel.Models import EncoderModels
import tensorflow as tf
# First Testing the Model without returning the attention weights
# bild the model
testModel=EncoderModels.Modeler(embedding_dim=8,
                                vocabulary_size=82,
                                conditional_string_length=100,
                                num_encoder_layer=4,num_heads=4,
                                num_neuron_pointwise=32,
                                return_attent_weights=False,
                                rate=0.1)
testModel.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
testTensor=tf.random.uniform(shape=(100,100),minval=0,maxval=82,
                                 dtype= tf.dtypes.int32)
# call the model to check the forward pass is working:
dumRes=testModel(testTensor,False)
print("Forward pass returned with a Tensor that has the following shape: "+
      str(dumRes.shape))
# print the summary of the model
testModel.summary()
## Test Model Training:
testModel.fit(x=testTensor,y=testTensor,batch_size=10)
## Evaluating the Models with return self attention weights:
testModel2=EncoderModels.Modeler(embedding_dim=8,
                                vocabulary_size=82,
                                conditional_string_length=100,
                                num_encoder_layer=1,num_heads=4,
                                num_neuron_pointwise=32,
                                return_attent_weights=True,
                                rate=0.1)
testModel2.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())

dumRes2,attent_weight=testModel2(testTensor,False)
print("Forward pass returned with an output Tensor that has the following shape: "+
      str(dumRes2.shape)+ "\n and and a list of self-attention matrices, the length"+
      "of the list is "+str(len(attent_weight))+" and the shape of the first matrix is"+
      str(attent_weight[0].shape))
# print the summary of the model
testModel2.summary()
## Test Model Training:
testModel2.fit(x=testTensor,y=testTensor,batch_size=10)