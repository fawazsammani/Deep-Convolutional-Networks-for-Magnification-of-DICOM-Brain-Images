import keras
from __future__ import print_function
import numpy as np
from keras.layers import Input, Convolution2D, concatenate, add, Activation
from keras.models import Model
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import h5py
import math
from keras import backend as K
from keras.layers.advanced_activations import PReLU

def PSNRLoss(y_true, y_pred):
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 20
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

batch_size = 64
nb_epoch = 100

#input imaage dimensions
img_rows, img_cols = 41, 41
out_rows, out_cols = 41, 41

##load train data
file = h5py.File('train.h5', 'r')
in_train = file['data'][:]
out_train = file['label'][:]
file.close()

#load validation data
file = h5py.File('test.h5', 'r')
in_test = file['data'][:]
out_test = file['label'][:]
file.close()

#convert data form 
in_train = in_train.astype('float32')
out_train = out_train.astype('float32')
in_test = in_test.astype('float32')
out_test = out_test.astype('float32')
in_train = in_train.reshape(in_train.shape[0], img_rows, img_cols,1)
in_test  = in_test.reshape(in_test.shape[0], img_rows, img_cols,1)
out_train = out_train.reshape(out_train.shape[0], out_rows, out_cols,1)
out_test = out_test.reshape(out_test.shape[0], out_rows, out_cols,1)
input_shape = (img_rows, img_cols,1)

#print number of training patches
print('in_train shape:', in_train.shape)
print(in_train.shape[0], 'train samples')
print(in_test.shape[0], 'test samples')
#Build the network
input_shape = (41,41,1)
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

model.summary()

adam = Adam(lr=0.001)
#compile
model.compile(loss='mse', metrics=[PSNRLoss], optimizer=adam)     
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
#Monitor the val_loss, and stop training if it dosen't improve for 5 times. #Note: This code was trained on 15 times
early_stopping_monitor = EarlyStopping(patience=5)
history = model.fit(in_train, out_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks = [lrate, early_stopping_monitor],
          verbose=1, validation_data=(in_test, out_test))            
print(history.history.keys())
#save model and weights
model.save('complete_network1.h5')
json_string = model.to_json()  
open('modelonly_network1.json','w').write(json_string)  
model.save_weights('network1_weights.h5') 
# summarize history for loss
plt.plot(history.history['PSNRLoss'])
plt.plot(history.history['val_PSNRLoss'])
plt.title('model loss')
plt.ylabel('PSNR/dB')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

