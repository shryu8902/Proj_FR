#%%
import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# os.chdir('./Proj_FR')
import tensorflow as tf
import tensorflow_probability as tfp
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
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

#%%
from sklearn.model_selection import KFold

train_ = np.load('/home/ashryu/Proj_FR/DATA/train_input.npy')
# val_ = np.load('/home/ashryu/Proj_FR/DATA/test_input.npy')[val_ind,...]
test_in_ = np.load('/home/ashryu/Proj_FR/DATA/test_input.npy')
test_out_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_input.npy')

train_input_ = train_[:,1:]
# val_input_ = val_[:,1:]
test_in_input_ = test_in_[:,1:]
test_out_input_ = test_out_[:,1:]

train_output_ = np.load('/home/ashryu/Proj_FR/DATA/train_output.npy')
# val_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[val_ind,...]
test_in_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')
test_out_output_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_output.npy')

kf = KFold(n_splits=5,shuffle=True,random_state=0)
kf.get_n_splits(train_)
# for train_index, test_index in kf.split(train_):
#     print(test_index[:10])

#%% LOAD SCALERS
with open('/home/ashryu/Proj_FR/DATA/IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
with open('/home/ashryu/Proj_FR/DATA/OUT_SCALER.pickle','rb') as f:
    OUT_SCALER = pickle.load(f)
#%%
SCALE = 'standard'
train_input = IN_SCALER[SCALE].transform(train_input_)
# val_input = IN_SCALER[SCALE].transform(val_input_)
test_in_input = IN_SCALER[SCALE].transform(test_in_input_)
test_out_input = IN_SCALER[SCALE].transform(test_out_input_)
train_output = OUT_SCALER[SCALE].transform(train_output_.reshape(-1,1)).reshape(-1,360)
# val_output = OUT_SCALER[SCALE].transform(val_output_.reshape(-1,1)).reshape(-1,360)
test_in_output = OUT_SCALER[SCALE].transform(test_in_output_.reshape(-1,1)).reshape(-1,360)
test_out_output = OUT_SCALER[SCALE].transform(test_out_output_.reshape(-1,1)).reshape(-1,360)

#%%

# Add positional embedding a. channelwise, b. just add to original data
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)
def diff_loss(y_true, y_pred):
    y_true_ = y_true[...,1:]- y_true[...,:-1]
    y_pred_ = y_pred[...,1:]- y_pred[...,:-1]
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    diff_loss = tf.keras.losses.MSE(y_true_,y_pred_)
    loss = mse_loss + 0.1*diff_loss
    return loss

def mae_diff_loss(y_true, y_pred):
    y_true_ = y_true[...,1:]- y_true[...,:-1]
    y_pred_ = y_pred[...,1:]- y_pred[...,:-1]
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    diff_loss = tf.keras.losses.MAE(y_true_,y_pred_)
    loss = mse_loss + 0.1*diff_loss
    return loss

def diff_loss_0(y_true,y_pred):
    y_true_ = y_true[...,1:]- y_true[...,:-1]
    y_pred_ = y_pred[...,1:]- y_pred[...,:-1]
    diff_loss = tf.keras.losses.MSE(y_true_,y_pred_)
    return diff_loss

def diff_loss_2(y_true, y_pred):
    y_true_ = y_true[...,1:]- y_true[...,:-1]
    y_pred_ = y_pred[...,1:]- y_pred[...,:-1]
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)

    for i in range(95,101):
        if i==95:
            diff_loss = tfp.stats.percentile(K.abs(y_true_-y_pred_),i,axis=1)
        else:
            diff_loss += tfp.stats.percentile(K.abs(y_true_-y_pred_),i,axis=1)
    # diff_loss = K.max(K.abs(y_true_-y_pred_),axis=1)
    # diff_loss = tf.keras.losses.MSE(y_true_,y_pred_)
    loss = mse_loss + 0.1*diff_loss
    return loss

def diff_loss_3(y_true, y_pred):
    y_true_ = y_true[...,1:]- y_true[...,:-1]
    y_pred_ = y_pred[...,1:]- y_pred[...,:-1]
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    diff_loss = tf.keras.losses.MSE(y_true_,y_pred_)
    loss = mse_loss + diff_loss
    return loss

def alpha_diff_loss(y_true,y_pred):
    y_true_ = y_true[...,1:]- y_true[...,:-1]
    y_pred_ = y_pred[...,1:]- y_pred[...,:-1]
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    diff_loss = tf.keras.losses.MSE(y_true_,y_pred_)
    loss = alpha*mse_loss + (1-alpha)*diff_loss
    return loss

#%%
from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten, LeakyReLU,Conv1D, PReLU, Dropout, Concatenate
from tensorflow.keras.models import Model

def RNN_model_v3(n_cell=256, n_layers=2, FN=[128,64], dr_rates=0.2, PE=None, PE_d = 16):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)
    if PE == 'add':
        pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
        inputs_extend = 0.5*pos_enc_tile+inputs_extend    
    elif PE == 'concat':
        pos_enc_tile = tf.tile(positional_encoding(360,PE_d), [num_rows, 1,1],name='pos_enc_tile')
        inputs_extend = Concatenate()([inputs_extend,pos_enc_tile])
    for i in range(n_layers):
        if i == 0:
            lstm = Bidirectional(LSTM(n_cell, return_sequences=True,dropout=dr_rates))(inputs_extend)       
        else:
            lstm = Bidirectional(LSTM(n_cell, return_sequences=True,dropout=dr_rates))(lstm)
    for i,j in enumerate(FN):
        if i ==0:
            FN_layer = TimeDistributed(Dense(j,activation='relu'))(lstm)
        else:
            FN_layer = TimeDistributed(Dense(j,activation='relu'))(FN_layer) 
    FN_drop = Dropout(dr_rates)(FN_layer)
    FN_out = TimeDistributed(Dense(1))(FN_drop)
    outputs = Flatten()(FN_out)
    model= Model(inputs, outputs)
    return model

#%%
for SEED, (train_index, val_index) in enumerate(kf.split(train_)):
    tf.random.set_seed(SEED)
    # vername = 'RNN-1-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None),MSE
    # er = 'mean_squared_error'
    # vername = 'RNN-2-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None),diff_loss
    # er = diff_loss
    # vername = 'RNN-3-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=add),MSE
    # er = 'mean_squared_error'
    # vername = 'RNN-4-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=8),MSE
    # er = 'mean_squared_error'
    # vername = 'RNN-5-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=8),diff_loss
    # er = diff_loss
    # vername = 'RNN-6-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),MSE
    # er = 'mean_squared_error'
    # vername = 'RNN-7-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),diff_loss
    # er = diff_loss
    # vername = 'RNN-8-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),diff_loss
    # er = mae_diff_loss
    # vername = 'RNN-9-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=8),diff_loss
    # er = mae_diff_loss
    # vername = 'RNN-10-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=8),diff_loss
    # er = diff_loss, sgd optimizer

    vername = 'kRNN-10/S{}'.format(SEED) 
    er = diff_loss
    base_model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE ='concat', PE_d=8)
    base_model.compile(optimizer='sgd', loss = er)

    ensure_dir('./MODEL/{}/'.format(vername))
    path = './MODEL/{}'.format(vername)+'/e{epoch:04d}.ckpt'
    checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
                save_best_only = True,
                mode = 'auto',
                save_weights_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    hist=base_model.fit(train_input[train_index,:],train_output[train_index,:],
                validation_data=(train_input[val_index,:],train_output[val_index,:]),
                callbacks=[checkpoint, reduce_lr, early_stopping],epochs=100,batch_size=256)

              