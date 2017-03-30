# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:16:04 2017

@author: Vishal Raman
"""

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense 
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from PIL import Image


    
seed=7
np.random.seed(seed)

(x_train,y_train),(x_test,y_test)=mnist.load_data()
#plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))
#print x_train[0]

numPixels=x_train.shape[1]*x_train.shape[2]
#print x_train.shape[0]
x_train=x_train.reshape(x_train.shape[0],numPixels).astype('float32')

x_test=x_test.reshape(x_test.shape[0],numPixels).astype('float32')
#print x_train.shape[0]

x_train=x_train/255
x_test=x_test/255


y_train=np_utils.to_categorical(y_train)

y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]


def create_model():
    model=Sequential()
    model.add(Dense(numPixels,input_dim=numPixels,init='normal',activation='relu'))
    model.add(Dense(num_classes,init='normal',activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics =['accuracy'] )
    return model
    
model=create_model()
model.fit(x_train,y_train,nb_epoch=10,batch_size=200 , verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)


model.save('digitRecognitionModel.h5')  # creates a HDF5 file 'my_model.h5'

#pre=x_test[0]

#==============================================================================
# pre=np.array([pre])
# #print pre.shape[0]
# predictedValue =model.predict(pre)
# print predictedValue
# rounded = [round(x) for x in predictedValue[0]]
# print rounded
# predictedClass=np_utils.probas_to_classes(predictedValue)
# print predictedClass
#==============================================================================

#######################prediction of image file#########################

originalImage = Image.open('digitsImages/5.bmp')
reducedImage = originalImage.resize((28,28))
#plt.imshow(reducedImage ,cmap=plt.get_cmap('gray'))
imageArray= np.array(reducedImage)
numPixels=imageArray.shape[0]*imageArray.shape[1]
imageArray=imageArray.reshape(1,numPixels).astype('float32')

predictedValue =model.predict(imageArray)
predictedClass=np_utils.probas_to_classes(predictedValue)
print "Digit is ",predictedClass


    
    