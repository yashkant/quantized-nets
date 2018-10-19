# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K
import numpy as np
import tensorflow as tf
from base_ops import round_through, _hard_sigmoid


def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    return 2 * round_through(_hard_sigmoid(x)) - 1


def binarize(W, H=1):
    Wb = H * binary_tanh(W / H)
    return Wb


def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = binarize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    return Wa, Wb


