#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:23:29 2018

@author: mpcr
"""

#git clone https://github.com/tensorflow/nmt/
#https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
#https://towardsdatascience.com/sequence-to-sequence-tutorial-4fde3ee798d8

## NMT seq2seq ##

#encoder_inputs [max_encoder_time, batch_size]: source input words.
#decoder_inputs [max_decoder_time, batch_size]: target input words.
#decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to the left by one time step with an end-of-sentence tag appended on the right.



import tensorflow as tf



#data pipeline#
#create dataset

#convert each sentence into vectors of word strings
dataset = dataset.map(lambda string: tf.string_split([string]).values)
#convert each sentence vector to tuple
dataset = dataset.map(lambda words: (words, tf.size(words))
#vocabualry lookup each sentence given a looup table object table (tuple-->int str)
dataset = dataset.map(lambda words, size: (table.lookup(words), size))

#hyperparameters
batch_size = 10
learning_rate = 0.001
 
#embedding (model learns the embedding)#
embedding_encoder = variable_scope.get_variable("embedding_encoder", [src_vocab_size, embedding_size], ...)
encoder_inputs: [max_time, batch_size]
encoder_emb_inp: [max_time, batch_size, embedding_size]
encoder_emb_inp = embedding_ops.embedding_lookup(embedding_encoder, encoder_inputs)   

#encoder for source language#
#encoder lstm
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
# Run Dynamic RNN --> accounts for varrying sequence lengths
encoder_outputs: [max_time, batch_size, num_units]
encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_sequence_length, time_major=True)

#decoder for target language#
#attention_states: [batch_size, max_time, num_units]
attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
#Create an attention mechanism
attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_states, memory_sequence_length=source_sequence_length)
#Decoder lstm
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
#Helper
helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_emb_inp, decoder_lengths, time_major=True)
#Decoder
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=num_units)
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
logits = outputs.rnn_output

#projection layer turns top hidden states to logit vectors of dimension V#
projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)

#loss#
#forward pass
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)
#backpropagation
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
#optimization
optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

#inference

#graphs





