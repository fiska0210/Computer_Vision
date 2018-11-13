import keras
import sys
import os
import cv2
import numpy as np
import keras
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras import backend as K
from keras import regularizers

def load_imgs():
    root = sys.argv[1]
    test_x = []
    ids = []
    for image in os.listdir(root):
        f = (root + '/' + image)
        test_x.append(cv2.imread(f, 0))
        ids.append(int(image[0:4]))

    test_x = np.array(test_x, dtype=np.float32)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
    test_x /= 255

    return ids, test_x

img_id,test_x=load_imgs()
model=load_model('save/Model.08-0.9909.hdf5')
y=model.predict(test_x)
y=np.argmax(y, axis=1)

with open (sys.argv[2],'w') as f:
    f.write('id,label\n')
    for i,label in enumerate(y):
        f.write('{},{}\n'.format(img_id[i],label))

        
