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
def softlabel(input_data):
    new_input = np.zeros_like(input_data)
    for i,matrix in enumerate(input_data):
        for j,vector in enumerate(matrix):
            k = np.convolve(vector,[0.05,0.1,0.7,0.1,0.05],'same')
            if np.sum(k)!=1:
                k=k/np.sum(k)
            new_input[i,j,...]=k
    return new_input

#%%
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

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

# train_output_ = np.load('/home/ashryu/Proj_FR/DATA/train_output.npy')
# val_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[val_ind,...]

# test_in_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[test_ind,...]
# test_out_output_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_output.npy')
train_output_prob = np.load('./DATA/soft_tr_out_prob.npy')
val_output_prob = np.load('./DATA/soft_val_out_prob.npy')
# test_in_output_prob = np.load('./DATA/te_in_out_prob.npy')
# vest_out_output_prob = np.load('./DATA/te_out_out_prob.npy')

# #%%
# from sklearn.preprocessing import KBinsDiscretizer
# NBIN=64

# discretizer = KBinsDiscretizer(n_bins=NBIN, encode='onehot-dense', strategy='quantile')
# discretizer.fit(train_output_.reshape(-1,1))
# with open('./MODEL/64qDiscretizer.pickle','wb') as f:
#     pickle.dump(discretizer,f)
#%%
# train_output_prob = discretizer.transform(train_output_.reshape(-1,1)).reshape(-1,360,NBIN)
# val_output_prob = discretizer.transform(val_output_.reshape(-1,1)).reshape(-1,360,NBIN)
# test_in_output_prob = discretizer.transform(test_in_output_.reshape(-1,1)).reshape(-1,360,NBIN)
# test_out_output_prob = discretizer.transform(test_out_output_.reshape(-1,1)).reshape(-1,360,NBIN)

# np.save('./DATA/tr_out_prob.npy',train_output_prob)
# np.save('./DATA/val_out_prob.npy',val_output_prob,)
# np.save('./DATA/te_in_out_prob.npy',test_in_output_prob)
# np.save('./DATA/te_out_out_prob.npy',test_out_output_prob)

# soft_train_out_prob = softlabel(train_output_prob)
# soft_val_out_prob = softlabel(val_output_prob)
# soft_test_in_out_prob = softlabel(test_in_output_prob)
# soft_test_out_out_prob = softlabel(test_out_output_prob)
# np.save('./DATA/soft_tr_out_prob.npy',soft_train_out_prob)
# np.save('./DATA/soft_val_out_prob.npy',soft_val_out_prob)
# np.save('./DATA/soft_te_in_out_prob.npy',soft_test_in_out_prob)
# np.save('./DATA/soft_te_out_out_prob.npy',soft_test_out_out_prob)



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
# train_output = OUT_SCALER[SCALE].transform(train_output_.reshape(-1,1)).reshape(-1,360)
# val_output = OUT_SCALER[SCALE].transform(val_output_.reshape(-1,1)).reshape(-1,360)
# test_in_output = OUT_SCALER[SCALE].transform(test_in_output_.reshape(-1,1)).reshape(-1,360)
# test_out_output = OUT_SCALER[SCALE].transform(test_out_output_.reshape(-1,1)).reshape(-1,360)

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

#%%
from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten, LeakyReLU,Conv1D, PReLU, Dropout, Concatenate, GRU
from tensorflow.keras.models import Model

def RNN_model_v3_cat(n_cell=256, n_layers=2, FN=[128,64], dr_rates=0.2, PE=None, PE_d = 16,n_bins=64):
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
    FN_out = TimeDistributed(Dense(n_bins,activation='softmax'))(FN_drop)
    outputs = FN_out
    model= Model(inputs, outputs)
    return model
#%%
for SEED in range(5):
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

    vername = 'eRNN-12/S{}'.format(SEED) 
    er = 'categorical_crossentropy'
    base_model = RNN_model_v3_cat(256,2,[128,64],dr_rates=0.2, PE='concat', PE_d=4)
    base_model.compile(optimizer='adam', loss = er)

    ensure_dir('./MODEL/{}/'.format(vername))
    path = './MODEL/{}'.format(vername)+'/e{epoch:04d}.ckpt'
    checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
                save_best_only = True,
                mode = 'auto',
                save_weights_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    hist=base_model.fit(train_input,train_output_prob,
                validation_data=(val_input,val_output_prob),
                callbacks=[checkpoint, reduce_lr, early_stopping],epochs=100,batch_size=256)
              
