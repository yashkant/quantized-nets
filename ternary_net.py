from __future__ import print_function
import numpy as np
import sys
seed = 1337
np.random.seed(seed) 

# ------------ Command Line Parameters ------------
import os
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
# -------------------------------------------------

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, Callback
from keras.utils import np_utils
from keras.activations import relu
from keras.callbacks import ModelCheckpoint
from ternary_ops import ternarize as ternarize_op
from ternary_layers import TernaryDense, TernaryConv2D
import mnist_data
import math


def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize_op(x)


def mnist_process(x):
	for j in range(len(x)):
		x[j] = x[j]*2-1
		if(len(x[j][0]) == 784):
			x[j] = np.reshape(x[j], [-1, 28, 28, 1])
	return x


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))



H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 128
epochs = 1000 
channels = 1
img_rows = 28 
img_cols = 28 
filters = 32 
kernel_size = (3, 3)
pool_size = (2, 2)
hidden_units = 128
classes = 10
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9


def add_conv_layer(model, conv_num_filters, conv_kernel_size, conv_strides, mpool_kernel_size, mpool_strides, ternarize):

    model.add(TernaryConv2D(conv_num_filters, kernel_size=(conv_kernel_size,conv_kernel_size), input_shape=( img_rows, img_cols, channels),
                           data_format='channels_last', strides=(conv_strides,conv_strides),
                           H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                           padding='valid', use_bias=use_bias, ternarize = ternarize))
    model.add(MaxPooling2D(pool_size=(mpool_kernel_size, mpool_kernel_size),strides = (mpool_strides,mpool_strides) ,padding='valid' , data_format='channels_last'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1))
    model.add(Activation(ternary_tanh))
    return model


# -------------Model Architecture--------------

model = Sequential()

#conv-layer 1 
conv_kernel_size = 3
conv_num_filters = 32
conv_strides = 2
mpool_kernel_size = 2
mpool_strides = 1
ternarize = True
add_conv_layer(model = model, conv_num_filters = conv_num_filters, conv_kernel_size = conv_kernel_size, conv_strides = conv_strides, mpool_kernel_size = mpool_kernel_size, mpool_strides = mpool_strides, ternarize = ternarize)


#conv-layer 2 
conv_kernel_size = 5
conv_num_filters = 256
conv_strides = 2
mpool_kernel_size = 2
mpool_strides = 2
ternarize = True
add_conv_layer(model = model, conv_num_filters = conv_num_filters, conv_kernel_size = conv_kernel_size, conv_strides = conv_strides, mpool_kernel_size = mpool_kernel_size, mpool_strides = mpool_strides, ternarize = ternarize)


# dense1 
model.add(Flatten())
model.add(TernaryDense(256, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
model.add(Activation(ternary_tanh))

# dense2
model.add(TernaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, ternarize = True))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()

# --------------------------------------------------


# ------------- MNIST Unpack and Augment Code------------

train_total_data, train_size, test_data, test_labels = mnist_data.prepare_MNIST_data(False)
train_data = train_total_data[:, :-10]
train_labels = train_total_data[:, -10:]

x = [train_data, train_labels, test_data, test_labels]
x_train, y_train, x_test, y_test = mnist_process(x)

print("X train: ", x_train.shape)
print("Y train: ", y_train.shape)

# --------------------------------------------------------


lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=[lr_scheduler, ModelCheckpoint('temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True)])
score = model.evaluate(x_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])