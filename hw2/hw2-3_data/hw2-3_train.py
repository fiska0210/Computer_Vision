import keras
import sys
import os
import cv2
import numpy as np
import sys

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras import backend as K
from keras import regularizers
from keras.utils import plot_model

import matplotlib.pyplot as plt

def load_imgs():
    
    train_x=[]
    train_y=[]
    root = sys.argv[1]
    for category in os.listdir(root + '/train'):
            print ('loading category {}...'.format(category))
            label = category[-1]
            for image in os.listdir('./train/' + category):
                f = ('./train/' + category + '/' + image)
                train_x.append(cv2.imread(f, 0))
                train_y.append(label)
                
    train_x=np.array(train_x,dtype=np.float32)
    train_x=train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
    train_x/=255
    train_y=np.array(train_y,dtype=np.float32)
    
    val_x=[]
    val_y=[]
    for category in os.listdir(root + '/valid'):
            label = category[-1]
            for image in os.listdir('./valid/' + category):
                f = ('./valid/' + category + '/' + image)
                val_x.append(cv2.imread(f,0))
                val_y.append(label)
                   
    val_x=np.array(train_x,dtype=np.float32)
    val_x=train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
    val_x/=255
    val_y=np.array(train_y,dtype=np.float32)
    
    return train_x,train_y,val_x,val_y

train_x,train_y,val_x,val_y = load_imgs()

(x_train, y_train), (x_test, y_test) = (train_x,train_y),(val_x,val_y)

import tensorflow as tf
import numpy      as np
import os
import argparse
import keras

from keras.models               import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.optimizers           import Adam, Adadelta
from keras.utils                import np_utils
from keras                      import regularizers
from keras.preprocessing.image  import ImageDataGenerator
from keras.layers               import Dense, Dropout, Activation, Flatten
from keras.layers               import Convolution2D, Conv2D, AveragePooling2D
from keras.layers               import ZeroPadding2D, MaxPooling2D
from keras.callbacks            import ModelCheckpoint,EarlyStopping

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print (x_train.shape)

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print (x_train.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
filepath='save/Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), 
         callbacks=callbacks_list)
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


