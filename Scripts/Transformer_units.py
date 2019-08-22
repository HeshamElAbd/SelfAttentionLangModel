#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:27:42 2019
@author: Hesham El Abd
@Description: Building Transformer units 
"""
## import modules:
import numpy as np
import tensorflow as tf
## First positional encoding: 
def get_angle(pos,i,d_model):
    angle_rates = 1/np.power(10000, (2*(i//2))/np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position,d_model):
    angle_rads=get_angle(np.arange(position)[:,np.newaxis],
                         np.arange(d_model)[np.newaxis,:],
                         d_model)
    angle_rads[:,0::2]=np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2]=np.cos(angle_rads[:,1::2])
    pos_encoding=angle_rads[np.newaxis,...]
    return tf.cast(pos_encoding,tf.float32)

def create_padding_mask(seq):
    seq=tf.cast(tf.math.equal(seq,0), tf.float32)
    return seq[:,tf.newaxis,tf.newaxis,:]

def scaled_dot_product_attention(q,k,v,mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

      # add the mask to the scaled tensor.
    if mask is not None:
          scaled_attention_logits += (mask * -1e9)  

 
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
        self.dense = tf.keras.layers.Dense(d_model)
        
        
    def split_heads(self,x,batch_size):
         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
         return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
    
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
    
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    
        scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v, mask)
    
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights
    
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation='relu'),  
                tf.keras.layers.Dense(d_model)  
                ])
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, return_attent_weights=True,
                 rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.return_attent_weights=return_attent_weights
    def call(self, x, training, mask):
        attn_output, atten_weights = self.mha(x, x, x, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  
    
        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  
        if not self.return_attent_weights:
            return out2
        else:
            return out2, atten_weights

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               seq_len,rate=0.1,return_attent_weights=True):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(seq_len, self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, return_attent_weights,
                                    rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
    self.return_attent_weights=return_attent_weights
        
  def call(self, x, training, mask):
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding

    x = self.dropout(x, training=training)
    if self.return_attent_weights==False:
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x
    else:
        attentWeights=[]
        for i in range(self.num_layers):
            x, attent_weights= self.enc_layers[i](x, training, mask)
            print(x,attent_weights)
            attentWeights.append(attent_weights)
        return x, attentWeights
        
    
class Modeler(tf.keras.layers.Layer):
    def __init__(self,embedding_dim,
                 vocabulary_size,
                 conditional_string_length,
                 num_encoder_layer,num_heads,dff,seq_len,rate=0.1,
                 return_attent_weights=True):
        """
        ## inputs:
        
        # embedding_dim: is the embeedind dimension to the model. 
        
        # vocabulary_size: is the number of words or tokens in the input 
        language to model.
        
        # conditional_string_length: is the input sequence length. 
        
        # num_encoder_layer: is the number of layers inside the encoder.
        
        # num_heads: is the number of heads inside the encoder, used for Multi-headed
        attention. 
        
        # rate: is the drop out rate. 
        """
        super(Modeler,self).__init__()
        self.return_attent_weights=return_attent_weights
        
        if return_attent_weights:
            self.encoder=Encoder(num_layers=num_encoder_layer,
                             d_model=embedding_dim, 
                             num_heads=num_heads, dff=dff, 
                             input_vocab_size=vocabulary_size,
                             seq_len=seq_len,
                             rate=rate,
                             return_attent_weights=return_attent_weights)
        else: 
            self.encoder=Encoder(num_layers=num_encoder_layer,
                             d_model=embedding_dim, 
                             num_heads=num_heads, dff=dff, 
                             input_vocab_size=vocabulary_size,
                             seq_len=seq_len,
                             rate=rate)
        self.pred_logits =tf.keras.layers.Dense(vocabulary_size)
        
    def call(self, x,training,mask):
        if not self.return_attent_weights:
            encoded_seq=self.encoder(x,training,mask)
            return self.pred_logits(encoded_seq)
        else: 
            encoded_seq,attent_weights=self.encoder(x,training,mask)
            return self.pred_logits(encoded_seq), attent_weights
        
        
        
        
        
        
        
        
        
        
        
        
        
        