#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from config import cfg
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
# tf.enable_eager_execution()
def mean_aggregator(node_features,neigh_features,out_dims,scope_name):
    #mean aggregator: σ(W*MEAN({hk−1} ∪ {hk−1, ∀u ∈ N (v)}).
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        if cfg.concat:
            node_embed = tf.expand_dims(node_features,1)
            to_feats = tf.concat([neigh_features,node_embed],1)
        else:
            to_feats = neigh_features
        combined = tf.reduce_mean(tf.layers.dense(to_feats,units=out_dims,activation=cfg.act),axis=1)
    return combined

def pooling_aggreagtor(node_features,neigh_features,out_dims,scope_name):
    #pooling aggregator: max({σ(W_pool*h^k_u + b) , ∀u_i ∈ N (v)})
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        if cfg.concat:
            node_embed = tf.expand_dims(node_features,1)
            to_feats = tf.concat([neigh_features,node_embed],1)
        else:
            to_feats = neigh_features
        combined = tf.reduce_max(tf.layers.dense(to_feats,units=out_dims,activation=cfg.act),axis=1)
    return combined

def lstm_aggregator(node_features,neigh_features,out_dims,scope_name):
    with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
        if cfg.concat:
            node_embed = tf.expand_dims(node_features,1)
            to_feats = tf.concat([neigh_features,node_embed],1)
        else:
            to_feats = neigh_features
        lstm = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(out_dims) for _ in range(1)], state_is_tuple = True)
        init_state = lstm.zero_state(tf.shape(to_feats)[0], dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(lstm, inputs=to_feats, initial_state=init_state, time_major=False)
        combined = state[-1][1]
    return combined














