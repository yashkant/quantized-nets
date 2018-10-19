# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K
from base_ops import switch




def _ternarize(W, H=1):
    W /= H
    ones = K.ones_like(W)
    zeros = K.zeros_like(W)
    Wt = switch(W > 0.5, ones, switch(W <= -0.5, -ones, zeros))
    Wt *= H
    return Wt


def ternarize(W, H=1):
    Wt = _ternarize(W, H)
    return W + K.stop_gradient(Wt - W)


def ternarize_dot(x, W):
    Wt = _ternarize(W)
    return K.dot(x, W) + K.stop_gradient(K.dot(x, Wt - W))