import keras
import sys
import os
import cv2
import numpy as np
import pdb
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras import backend as K
from keras import regularizers
from keras.models import load_model
import cv2
#import matplotlib.pyplot as plt

nth_filter=int(sys.argv[1])
#layer_name='conv2d_3'
layer_name='conv2d_4'

model=load_model('save/Model.08-0.9909.hdf5')
model.summary()
layer_dict=dict([(layer.name,layer) for layer in model.layers])

input_img=model.input
layer=layer_dict[layer_name]
#conv_2=layer_dict['conv2d_2']


print (layer.output)
loss_1=K.mean(layer.output[:,:,:,nth_filter])
grads=K.gradients(loss_1,input_img)[0]
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate=K.function([input_img],[loss_1, grads])
#input_img_data=np.random.random((1,28,28,1))*0.2+0.5
input_img_data=np.random.random((1,28,28,1))

#pdb.set_trace()
for j in range(1000):
    loss_value,grads_value=iterate([input_img_data])
    while (loss_value==0):
        input_img_data=np.random.random((1,28,28,1))*0.2+0.5
        loss_value,grads_value=iterate([input_img_data])
    input_img_data+=grads_value*10
    #print (loss_value,grads_value)
    #print (np.mean(input_img_data))
    input_img_data[input_img_data>1]=1
    input_img_data[input_img_data<0]=0

input_img_data=input_img_data.squeeze()
input_img_data*=255
input_img_data=input_img_data.astype(np.uint8)

cv2.imwrite('visual_img/{}_{}.png'.format(layer_name,nth_filter),input_img_data)
