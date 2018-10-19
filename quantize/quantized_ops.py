# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K
import numpy as np
import tensorflow as tf
from base_ops import round_through, pow_through, log2_through, _hard_sigmoid



# Method Broke! 
def log_quantize(W, nb = 16):
    non_sign_bits = nb
    W = tf.Print(W,[W, nb],message = "---- W Log Quantize normal---- \n ",summarize=50, first_n = 5)
    W_where = tf.where(W > 0, W, tf.zeros_like(W) +1e-15)
    W_where = tf.Print(W_where,[W_where, nb],message = "---- W Log Quantize after where---- \n ",summarize=50, first_n = 5)
    W_log = log2_through(W_where)
    W_log = tf.Print(W_log,[W_log, nb],message = "---- W Log Quantize after log--- \n ",summarize=50, first_n = 5)
    W_round = round_through(W_log)
    W_round = tf.Print(W_round,[W_round, nb],message = "---- W Log Quantize after round---- \n ",summarize=50, first_n = 5)
    W_clip = clip_through(W_round,0, nb-1)
    W_clip = tf.Print(W_clip,[W_clip, nb],message = "---- W Log Quantize after clip---- \n ",summarize=50, first_n = 5)
    W_pow = pow_through(W_clip, 2) 
    W_pow = tf.Print(W_pow,[W_pow, nb],message = "---- W Log Quantize after pow---- \n ",summarize=50, first_n = 5)
    W_fix = tf.where(W_round<0, tf.zeros_like(W_pow), W_pow)
    W_fix = tf.Print(W_fix,[W_fix, nb],message = "---- W Log Quantize after fix---- \n ",summarize=50, first_n = 5)
    return W_fix



def quantized_relu(W, nb=16):
    nb_bits = nb
    Wq = K.clip(2. * (round_through(_hard_sigmoid(W) * pow(2, nb_bits)) / pow(2, nb_bits)) - 1., 0,
                1 - 1.0 / pow(2, nb_bits - 1))
    # Wq = tf.Print(Wq,[Wq],message = "---- W Quantize Relu After---- \n ",summarize=5, first_n = 5)
    return Wq



def quantized_tanh(W, nb=16):
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = K.clip(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq

def quantize(W, nb = 16, clip_through=False):
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    W = tf.Print(W,[W, non_sign_bits],message = "---- W Quantize Before---- \n ",summarize=10, first_n = 5)
    if clip_through:
        Wq = clip_through(round_through(W*m),-m,m-1)/m
    else:
        Wq = K.clip(round_through(W*m),-m,m-1)/m
    Wq = tf.Print(Wq,[Wq],message = "---- Wq Quantize After---- \n ",summarize=10, first_n = 5)
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq