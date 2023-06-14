from keras.preprocessing.image import ImageDataGenerator
import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from keras import optimizers
from segment import segment_characters
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
import keras.backend as K

from keras.utils import get_custom_objects
char = segment_characters()
def val_acc(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    return K.mean(K.equal(y_true, y_pred))
def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c
    output = []
    model = load_model('/home/vishnu/Documents/ANMR_GCID/mymodel.h5')

    
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28))
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        get_custom_objects().update({'val_acc': val_acc})
        y_ = model.predict_step(img)[0] #predicting the class
        # key = np.array2string(y_)
        y_ref=y_.ref()
        character = dic[y_ref] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

x=show_results()
print(x)
