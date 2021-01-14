#%%
import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# os.chdir('./Proj_FR')
import tensorflow as tf
import numpy as np
import pandas as pd
import tqdm, glob, pickle, datetime, re, time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
import setGPU
import GAN_BASE
import pickle

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def MAPE(y_true,y_pred):
    ape = abs(y_true-y_pred)/y_true*100
    mape = np.mean(ape,axis=-1)
    return mape
def MSE(y_true,y_pred):
    se = (y_true-y_pred)**2
    mse = np.mean(se,axis=-1)
    return mse
#%%
# physical_devices = tf.config.list_physical_devices('GPU') 
# for gpu_instance in physical_devices: 
#     tf.config.experimental.set_memory_growth(gpu_instance, True)

import random as python_random

np.random.seed(0)
python_random.seed(0)
#%%
val_ind,test_ind = train_test_split(range(25988),test_size=18000, random_state=0)

train_ = np.load('/home/ashryu/Proj_FR/DATA/train_input.npy')
val_ = np.load('/home/ashryu/Proj_FR/DATA/test_input.npy')[val_ind,...]
# test_in_ = np.load('/home/ashryu/Proj_FR/DATA/test_input.npy')[test_ind,...]
# test_out_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_input.npy')

train_input_ = train_[:,1:]
val_input_ = val_[:,1:]
# test_in_input_ = test_in_[:,1:]
# test_out_input_ = test_out_[:,1:]

train_output_ = np.load('/home/ashryu/Proj_FR/DATA/train_output.npy')
val_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[val_ind,...]
# test_in_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[test_ind,...]
# test_out_output_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_output.npy')
#%% SAVE and LOAD SCALERS
# # SAVE SCALER
# OUT_SCALER = {'standard':StandardScaler(),'minmax':MinMaxScaler()}
# IN_SCALER = {'standard':StandardScaler(),'minmax':MinMaxScaler()}
# for dat in [train_input_,val_input_,test_in_input_,test_out_input_]:
#     IN_SCALER['standard'].partial_fit(dat)
#     IN_SCALER['minmax'].partial_fit(dat)

# for dat in [train_output_,val_output_,test_in_output_,test_out_output_]:
#     OUT_SCALER['standard'].partial_fit(dat.reshape(-1,1))
#     OUT_SCALER['minmax'].partial_fit(dat.reshape(-1,1))

# with open('./DATA/new_IN_SCALER.pickle','wb') as f:
#     pickle.dump(IN_SCALER,f)
# with open('./DATA/new_OUT_SCALER.pickle','wb') as f:
#     pickle.dump(OUT_SCALER,f)

# LOAD SCALER
# with open('/home/ashryu/Proj_FR/DATA/IN_SCALER.pickle','rb') as f: # scikit-learn 꼬이기 전
with open('./DATA/new_IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
# with open('/home/ashryu/Proj_FR/DATA/OUT_SCALER.pickle','rb') as f:# scikit-learn 꼬이기 전
with open('./DATA/new_OUT_SCALER.pickle','rb') as f:
    OUT_SCALER = pickle.load(f)

#%%
SCALE = 'standard'
train_input = IN_SCALER[SCALE].transform(train_input_)
val_input = IN_SCALER[SCALE].transform(val_input_)
# test_in_input = IN_SCALER[SCALE].transform(test_in_input_)
# test_out_input = IN_SCALER[SCALE].transform(test_out_input_)
train_output = OUT_SCALER[SCALE].transform(train_output_.reshape(-1,1)).reshape(-1,360)
val_output = OUT_SCALER[SCALE].transform(val_output_.reshape(-1,1)).reshape(-1,360)
# test_in_output = OUT_SCALER[SCALE].transform(test_in_output_.reshape(-1,1)).reshape(-1,360)
# test_out_output = OUT_SCALER[SCALE].transform(test_out_output_.reshape(-1,1)).reshape(-1,360)
#%%
with open('./DATA/kmeans.pkl','rb') as f:
    km_labels = pickle.load(f)
# specific class 0,5,6,9
new_label = [0 if x in [0,5,6,9] else 1 for x in km_labels[0]]
indexes = np.where(np.array(new_label)==0)[0]
aug_train_output_p1 = train_output[indexes,...]
aug_train_output_n1 = train_output[indexes,...]
# aug_train_output_p1 = np.append(train_output[:,1:],train_output[:,-1].reshape(-1,1),axis=1) 
# aug_train_output_n1 = np.append(train_output[:,0].reshape(-1,1),train_output[:,:-1],axis=1)
aug_train_output = np.concatenate((aug_train_output_p1,train_output,aug_train_output_n1))
aug_train_input = np.concatenate((train_input[indexes,:],train_input,train_input[indexes,:]))
#%%

#%%
for SEED in range(1):
    tf.random.set_seed(SEED)
    vername = 'SRGAN/S{}'.format(SEED) 
    base_model = GAN_BASE.SurroGAN(256,2,[128,64],0.2,'concat',4)
    # base_model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='concat', PE_d=4)
    # base_model.compile(optimizer='adam', loss = er, metrics=['mean_squared_error'])
    base_model.compile(metrics=['mean_squared_error','mean_absolute_error'],
                    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                    )
    ensure_dir('./MODEL/{}/'.format(vername))
    path = './MODEL/{}'.format(vername)+'/e{epoch:04d}.ckpt'
    checkpoint = ModelCheckpoint(path, monitor = 'loss',verbose = 1,
                save_best_only = True,
                mode = 'auto',
                save_weights_only = True)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, min_lr=1e-5)
    # early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    #hist= base_model.fit(train_input,train_output,   일반
    # hist=base_model.fit(np.tile(train_input,(3,1)),aug_train_output,     aug
    hist = base_model.fit(train_input, train_output,
                validation_data=(val_input,val_output),
                callbacks=[checkpoint, reduce_lr, early_stopping],epochs=100,batch_size=256)  ## epcoh 150
              
# %%
