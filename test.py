import keras
import numpy as np
from keras.layers import Input, Convolution2D, Activation 
from keras.layers import concatenate
from keras.layers import add
from keras.models import Model
from keras import backend as K
from keras.layers.advanced_activations import PReLU
import h5py
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave
import glob
import os
import re
from PIL import Image
from scipy.misc import imread, imsave, imresize
from scipy import misc

input_shape = (None,None,1)
x = Input(shape = input_shape)
c1 = Convolution2D(64, (3,3), init = 'he_normal', padding='same', name='Initial_Conv1')(x)
c1 = PReLU(shared_axes = [1,2])(c1)
c2 = Convolution2D(64, (3,3), init = 'he_normal', padding='same', name='Initial_Conv2')(c1)
c2 = PReLU(shared_axes = [1,2])(c2)
c3 = Convolution2D(64, (3,3), init = 'he_normal', padding='same', name='Initial_Conv3')(c2)
c3 = PReLU(shared_axes = [1,2])(c3)
c4 = Convolution2D(64, (3,3), init = 'he_normal', padding='same', name='Initial_Conv4')(c3)
c4 = PReLU(shared_axes = [1,2])(c4)
con = keras.layers.concatenate([x,c1,c2,c3,c4], axis=3)

"""
Residual Model
"""
modeli = Convolution2D(64, (1,1),  activation='relu', init = 'he_normal', padding='same', name='Res_Conv_1_1')(con)
modelb = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='Res_Conv_1_2')(modeli)
modelb = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='Res_Conv_1_3')(modelb)
res1 = modelb
out = add([res1, modeli])
modelc = Activation('relu')(out)

modelc = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='Res_Conv_2_1')(modelc)
modelc = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='Res_Conv_2_2')(modelc)
modelc = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='Res_Conv_2_3')(modelc)
res2 = modelc
out2 = add([res2, out])
modeld = Activation('relu')(out2)

modeld = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name = 'Res_Conv_3_1')(modeld)
modeld = Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name = 'Res_Conv_3_2')(modeld)
modeld = Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name = 'Res_Conv_3_3')(modeld)
res3 = modeld
out3 = add([res3,out2])
out3 = Activation('relu')(out3)

final_residual = Convolution2D(input_shape[2], (1, 1), padding='same', kernel_initializer='he_normal', name = 'Final_Residual')(out3)

"""
3 Towers Model 
"""
tower1 = Convolution2D(32, (1,1), activation='relu' , init = 'he_normal', padding='same', name='Tower1')(con)

tower2 = Convolution2D(32, (1,1), activation='relu' , init = 'he_normal', padding='same', name='Tower2_1')(con)
tower2 = Convolution2D(32, (3,3), activation='relu' , init = 'he_normal', padding='same', name='Tower2_2')(tower2)
tower2 = Convolution2D(32, (3,3), activation='relu' , init = 'he_normal', padding='same', name='Tower2_3')(tower2)
tower2 = Convolution2D(32, (3,3), activation='relu' , init = 'he_normal', padding='same', name='Tower2_4')(tower2)

tower3 = Convolution2D(32, (1,1), activation='relu' , init = 'he_normal', padding='same', name='Tower3_1')(con)
tower3 = Convolution2D(32, (5,5), activation='relu' , init = 'he_normal', padding='same', name='Tower3_2')(tower3)
tower3 = Convolution2D(32, (5,5), activation='relu' , init = 'he_normal', padding='same', name='Tower3_3')(tower3)
tower3 = Convolution2D(32, (5,5), activation='relu' , init = 'he_normal', padding='same', name='Tower3_4')(tower3)
finalcon = keras.layers.concatenate([tower1,tower2,tower3], axis=3)

finalconv = Convolution2D(input_shape[2], (1,1), init = 'he_normal', padding='same', name='Final_Tower_Conv')(finalcon)

alltog = keras.layers.concatenate([finalconv, final_residual])
end = Convolution2D(input_shape[2], (1,1), init = 'he_normal', padding='same', name='FinalConv')(alltog)

output_img = keras.layers.add([x, end])
model = Model(x, output_img)

model.load_weights('network1_weights.h5')

"""
Multiple Predictions with reading the images in order
"""
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

alist=[]
for filename in glob.glob('testing images_grayscale/*.png'):
    alist.append(filename)

alist.sort(key=natural_keys)

i=1
for file in alist:
    img = imread(file)
    print(img.dtype, img.shape) 
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)      
    img = img/255 
    img_pred = model.predict(img)  
    test_img = np.reshape(img_pred, (img.shape[1], img.shape[1]))
    imsave(str(i) + '.png', test_img)
    i=i+1

"""
Single Prediction
"""
img = imread('test.png')
print(img.dtype, img.shape) 
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)   
img = img/255 
img_pred = model.predict(img)  
test_img = np.reshape(img_pred, (img.shape[1], img.shape[1]))
imsave(str(1) + '.png', test_img) 

