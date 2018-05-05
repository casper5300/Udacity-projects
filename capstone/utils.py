
# coding: utf-8

# In[1]:


# load required packages
from keras.applications import *
from keras.models import *
from keras.preprocessing.image import *
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Dropout,BatchNormalization
import h5py as h
import numpy as np
from PIL import Image
import cv2
import os
import glob
import shutil
from keras import backend as K
from sklearn.utils import shuffle
from keras.utils import to_categorical
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
get_ipython().magic('matplotlib inline')





def get_file_num(path,name):
    return os.listdir(os.path.join(path,name))

def preprocess_data(arg):
    gen = ImageDataGenerator(**arg)
    return gen
    
def get_data(gen,path,input_size,batch_size):
    generator = gen.flow_from_directory(path,shuffle = True,target_size=(input_size,input_size),batch_size=batch_size)
    return generator

def get_test_data(gen,path,input_size,batch_size):
    generator = gen.flow_from_directory(path,shuffle = False,target_size=(input_size,input_size),batch_size=batch_size,class_mode=None)
    return generator

def comput_step(num_sample,batch_size):
    if num_sample%batch_size == 0:
        step = num_sample//batch_size 
    else:
        step = num_sample//batch_size+1
    return step   

# In[ ]:
def Fine_Tune(base_model,num):
    base_model.layers.pop()
    for layer in base_model.layers:
        layer.trainable=False
    x = base_model.output
    pred = Dense(num,activation = 'softmax')
    return Model(inputs=x,outputs=pred)    

def feature_extract(b_model,input_size,path,batch_size,pre_func):
    
    #preprocess data and loading       
    
    gen = ImageDataGenerator(preprocessing_function=pre_func)
    trn_generator = get_data(gen,path=path+'train',input_size=input_size,batch_size=batch_size)
    test_generator = get_test_data(gen,path=path+'test',input_size=input_size,batch_size=batch_size)
    
    #create pre-trained model
    
    base_model = b_model(weights='imagenet',include_top=False)
    model = Model(inputs = base_model.input,outputs=GlobalAveragePooling2D(base_model.output))
                  
    # extract features
    trn_features = model.predict_generator(trn_generator)
    test_features = model.predict_generator(test_generator)
                  
    # save features and labels
    
    h.File(str(base_model.name)+'_features').create_dataset('trn_features',data=trn_features)
    h.File(str(base_model.name)+'_features').create_dataset('test_featuers',data=test_features)
    h.File(str(base_model.name)+'_features').create_dataset('label',data=trn_generator.classes)
                  
                  
    return features
    
def load_features(filenames):
    x_train = []
    y_train = []
    x_test = []
    for fn in filenames:       

        with h.File(fn,'r') as f:
            x_train.append(np.array(f['trn_features']))
            x_test.append(np.array(f['test_featuers']))
            y_train.append(np.array(f['trn_label']))
        
    x_train = np.concatenate(x_train,axis=1)
    x_test = np.concatenate(x_test,axis=1)
    y_train = to_categorical(y_train,2)
    x_train,y_train = shuffle(x_train,y_train)
    
    return x_train,y_train,x_test
       
