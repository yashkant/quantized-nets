import keras
from keras.datasets import mnist
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import sys
import mnist_data
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, Callback, ReduceLROnPlateau
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
from quantize.quantized_layers import QuantizedConv2D, QuantizedDense
from quantize.quantized_ops import quantized_relu as quantized_relu_op
from quantize.quantized_ops import quantized_tanh as quantized_tanh_op

# ------------ Command Line Parameters ------------
import os
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
# -------------------------------------------------

num_classes = 10
epochs = 200
batch_size = 64

img_rows, img_cols = 28, 28

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# # ------------- MNIST Unpack and Augment Code------------

train_total_data, train_size, test_data, test_labels = mnist_data.prepare_MNIST_data(False)
train_data = train_total_data[:, :-10]
train_labels = train_total_data[:, -10:]

# x = [train_data, train_labels, test_data, test_labels]
x_train, y_train, x_test, y_test = train_data, train_labels, test_data, test_labels


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("X train: ", x_train.shape)
print("Y train: ", y_train.shape)

# # --------------------------------------------------------


# BN
epsilon = 1e-6
momentum = 0.9

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (img_rows, img_cols, 1), activation = 'relu', strides = 2))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 1))
# model.add(Dropout(0.25))

model.add(Conv2D(256, (5,5), activation = 'relu', strides=2))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
# model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(units = 256, activation = 'relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))

# model.add(Dropout(0.5))

model.add(Dense(units = num_classes, activation = 'softmax'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))

# lr & opt 1
opt = Adam(lr=lr_start) 
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)

# lr & opt 2 
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()



history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=[lr_scheduler, ModelCheckpoint('temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True)])


# model.fit(x_train, y_train, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
    
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])