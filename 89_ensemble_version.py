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

import random as python_random
np.random.seed(0)
python_random.seed(0)
session_conf = tf.ConfigProto(intra_op_parallelism_threads =1,
                                inter_op_parallelism_trheads = 1 )
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
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
#%% LOAD SCALERS
with open('/home/ashryu/Proj_FR/DATA/IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
with open('/home/ashryu/Proj_FR/DATA/OUT_SCALER.pickle','rb') as f:
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
from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten, LeakyReLU,Conv1D, PReLU, Dropout, Concatenate, GRU
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

def GRU_model_v3(n_cell=256, n_layers=2, FN=[128,64], dr_rates=0.2, PE=None, PE_d = 16):
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
            lstm = Bidirectional(GRU(n_cell, return_sequences=True,dropout=dr_rates))(inputs_extend)       
        else:
            lstm = Bidirectional(GRU(n_cell, return_sequences=True,dropout=dr_rates))(lstm)
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
for SEED in range(1):
    tf.random.set_seed(SEED)
    # vername = 'eRNN-1-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None),MSE
    # er = 'mean_squared_error' == RNN-1-S
    # vername = 'eRNN-2-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None),diff_loss
    # er = diff_loss  == RNN-2-S
    # vername = 'eRNN-3-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None),mae_diff_loss
    # er = mae_diff_loss
    # vername = 'eRNN-4-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=add),MSE
    # er = 'mean_squared_error' == RNN-3
    # vername = 'eRNN-5-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),MSE
    # er = 'mean_squared_error'
    # vername = 'eRNN-6-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),diff_loss
    # er = diff_loss
    # vername = 'eRNN-7-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),mae_diff_loss
    # er = mae_diff_loss
    # vername = 'eRNN-8-S{}'.format(SEED) #GRU_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),MSE
    # er = 'mean_squared_error'
    # vername = 'eRNN-9-S{}'.format(SEED) #GRU_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),diff_loss
    # er = diff_loss
    # vername = 'eRNN-10-S{}'.format(SEED) #GRU_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),mae_diff_loss
    # er = mae_diff_loss
    # vername = 'eRNN-11-S{}'.format(SEED) #RNN_model_v3_cat(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),'categorical_crossentropy',
    # er = 'categorical_crossentropy'
    # vername = 'eRNN-12-S{}'.format(SEED) #RNN_model_v3_cat(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),'categorical_crossentropy',
    # er = 'categorical_crossentropy', label smoothing
    # vername = 'eRNN-13-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),MSE
    # er = 'mean_squared_error', aug t+1, t-1 이동시켜서
    # vername = 'eRNN-14-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),diff_loss
    # er = diff_loss, aug aug t+1, t-1 이동시켜서
    # vername = 'eRNN-15-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),mae_diff_loss
    # er = mae_diff_loss,aug t+1, t-1 이동시켜서
    # vername = 'eRNN-16-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),MSE
    # er = 'mean_squared_error', aug(oversampling), and 150 epoch
    # vername = 'eRNN-17-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),diff_loss
    # er = diff_loss, aug(oversampling), and 150 epoch
    # vername = 'eRNN-18-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),mae_diff_loss
    # er = mae_diff_loss, aug(oversampling), and 150 epoch
    # vername = 'eRNN-19-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),MSE
    # er = 'mean_squared_error',aug(oversampling) val_loss 대신 val_mean_squared_error, epoch 150
    # vername = 'eRNN-20-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),diff_loss
    # er = diff_loss,aug(oversampling), val_loss 대신 val_mean_squared_error, epoch 150
    # vername = 'eRNN-21-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),mae_diff_loss
    # er = mae_diff_loss,aug(oversampling), val_loss 대신 val_mean_squared_error, epoch 150
    # vername = 'eRNN-22-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),diff_loss
    # er = diff_loss,aug(oversampling), val_loss, epoch 150
    # vername = 'eRNN-23-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),mae_diff_loss
    # er = mae_diff_loss,aug(oversampling), val_loss, epoch 150
    # vername = 'eRNN-30-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),mae_diff_loss
    # er = mae_diff_loss,aug(oversampling), val_loss, epoch 150
    # vername = 'eRNN-31-S{}'.format(SEED) #RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=concat,PE_d=4),mae_diff_loss
    # er = mae_diff_loss,aug(oversampling), val_loss, epoch 150

    vername = 'eRNN-31/S{}'.format(SEED) 
    er = diff_loss
    base_model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='concat', PE_d=4)
    base_model.compile(optimizer='adam', loss = er, metrics=['mean_squared_error'])

    ensure_dir('./MODEL/{}/'.format(vername))
    path = './MODEL/{}'.format(vername)+'/e{epoch:04d}.ckpt'
    checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
                save_best_only = True,
                mode = 'auto',
                save_weights_only = True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    #hist= base_model.fit(train_input,train_output,   일반
    # hist=base_model.fit(np.tile(train_input,(3,1)),aug_train_output,     aug
    hist = base_model.fit(aug_train_input, aug_train_output,
                validation_data=(val_input,val_output),
                callbacks=[checkpoint, reduce_lr, early_stopping],epochs=10,batch_size=256)  ## epcoh 150
              
# %%
