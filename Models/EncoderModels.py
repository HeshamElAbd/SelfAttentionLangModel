#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:51:15 2019

@author: Hesham El Abd
@Description: The Module uses the Encoder unit of the transformer to construct
two Types of models the first is the Modeler which is used to construct 
language models and the second is the annotator which is used to map an input
sequene to a specific numerical label.
"""
# loadthe modules
from SelfAttentionLangModel.Parts import EncoderParts
import tensorflow as tf

# define the models
class Modeler(tf.keras.Model):
    def __init__(self,embedding_dim,
                 vocabulary_size,
                 conditional_string_length,
                 num_encoder_layer,num_heads,
                 num_neuron_pointwise,
                 rate=0.1,
                 return_attent_weights=True):
        """
        The Modeler is a model that is used to construct language models.
        ## inputs:
        
        # embedding_dim: is the embeedind dimension for each input integer to 
        the model.
        
        # vocabulary_size: is the number of unique words, characters or tokens 
        in the language.i.e. input language.
        
        # conditional_string_length: is the input sequence length.
        
        # num_encoder_layer: is the number of layers inside the encoder.
        
        # num_heads: is the number of heads inside the encoder, used for
        Multi-headed attention. which is used to increase the power of the 
        model to learn from different representaional spaces. 
        
        # num_neuron_pointwise: is the number of neurons in the feed-forward
        point wise attention neuros.
        
        # rate: is the drop out rate. 
        """
        super(Modeler,self).__init__()
        
        self.return_attent_weights=return_attent_weights
        
        self.encoder=EncoderParts.Encoder(num_layers=num_encoder_layer,
                             d_model=embedding_dim, 
                             num_heads=num_heads, 
                             dff=num_neuron_pointwise, 
                             input_vocab_size=vocabulary_size,
                             seq_len=conditional_string_length,
                             rate=rate,
                             return_attent_weights=return_attent_weights)
        
        self.pred_logits =tf.keras.layers.Dense(vocabulary_size)
        
    def call(self, x,training):
        mask=EncoderParts.create_padding_mask(x)
        if not self.return_attent_weights:
            encoded_seq=self.encoder(x,training,mask)
            modelPredictionLogit=self.pred_logits(encoded_seq)
            return modelPredictionLogit
        else: 
            encoded_seq, attent_weights=self.encoder(x,training,mask)
            modelPredictionLogit=self.pred_logits(encoded_seq)
            return self.pred_logits(encoded_seq), attent_weights