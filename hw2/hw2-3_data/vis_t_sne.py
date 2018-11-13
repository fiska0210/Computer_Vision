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
from keras.models import Model,load_model
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

def load_imgs(base_dir):
    train_base_dir=os.path.join(base_dir,'train')
    train_data_dir=[os.path.join(train_base_dir,'class_{}'.format(i)) for i in range(10)]

    valid_base_dir=os.path.join(base_dir,'valid')
    valid_data_dir=[os.path.join(valid_base_dir,'class_{}'.format(i)) for i in range(10)]

    train_img_files=[]
    train_x=[]
    train_y=[]
    for i,train_dir in enumerate(train_data_dir):
        train_img_files=train_img_files+[os.path.join(train_dir,f) for f in os.listdir(train_dir) if (f[-4:]=='.png')]
        train_y=train_y+[i for f in os.listdir(train_dir) if (f[-4:]=='.png')]

    train_x=[cv2.imread(f,0) for f in train_img_files]
    train_x=np.array(train_x,dtype=np.float32)
    train_x=train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
    train_x/=255


    val_img_files=[]
    val_x=[]
    val_y=[]
    for i,valid_dir in enumerate(valid_data_dir):
        val_img_files=val_img_files+[os.path.join(valid_dir,f) for i,f in enumerate(os.listdir(valid_dir)) if (f[-4:]=='.png' and i<100)]
        val_y=val_y+[i for j,f in enumerate(os.listdir(valid_dir)) if (f[-4:]=='.png' and j<100)]
    val_x=[cv2.imread(f,0) for f in val_img_files]
    val_x=np.array(val_x,dtype=np.float32)
    val_x=val_x.reshape(val_x.shape[0],val_x.shape[1],val_x.shape[2],1)
    val_x/=255
    return train_x,train_y,val_x,val_y

base_dir='./'
train_x,train_y,val_x,val_y=load_imgs(base_dir)
del train_x, train_y
print (val_x.shape)
print (len(val_y))

layer_name='conv2d_4'

model=load_model('save/Model.08-0.9909.hdf5')
model.summary()
#pdb.set_trace()
intermediate_layer_model=Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
intermediate_output=intermediate_layer_model.predict(val_x)
intermediate_output=intermediate_output.reshape(intermediate_output.shape[0],-1)
print (intermediate_output.shape)

embedded=TSNE(n_components=2).fit_transform(intermediate_output)
print (embedded.shape)

color=['b','g','r','c','m','y','k','burlywood','chartreuse','gray']
#pdb.set_trace()
#for i in range(10):
for i in range(embedded.shape[0]):
    print (i)
    plt.plot(embedded[i,0],embedded[i,1],color=color[val_y[i]],marker='o')
plt.savefig('{}_tsne.png'.format(layer_name))

