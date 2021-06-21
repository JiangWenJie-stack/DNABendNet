from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
import pandas as pd
from load2data import extraData
import random
import os
import random
import torch

data = pd.read_csv("D:\JWJ\PycharmWorkplace\Data\\no_repeat_alldata.csv",sep = ",")
data_frame = np.array(data)

random.shuffle(data_frame)
np.random.shuffle(data_frame)

data = pd.DataFrame(data_frame)
data_x= np.array(data.iloc[:,7])
data_y = np.array(data.iloc[:,4])
def hot(chain):
    b = np.zeros([len(chain),4])
    for i in range(len(chain)):
        if chain[i]=='A':
            b[i,0]=1
        elif chain[i]=='T':
            b[i,1]=1
        elif chain[i]=='G':
            b[i,2]=1
        elif chain[i]=='C':
            b[i,3]=1
    return b

for i in range(len(data)):
    data_x[i]=hot(data_x[i])

Data_x=np.ones(len(data)*50*4).reshape(len(data),50,4)
for i in range(len(data)):
    Data_x[i]=data_x[i]
Data_x = Data_x.reshape(Data_x.shape[0],Data_x.shape[1],Data_x.shape[2],1)

X_train = np.array(Data_x[:int(len(data)*0.6)])
Y_train = np.array(data_y[:int(len(data)*0.6)])
for i in range(len(Y_train)):
    Y_train[i] = float(Y_train[i])

X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).shuffle(40000).batch(128)







