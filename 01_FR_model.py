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
val_ind,test_ind = train_test_split(range(25988),test_size=18000, random_state=0)
train_ = np.load('./DATA/train_input.npy')
val_ = np.load('./DATA/test_input.npy')[val_ind,...]
test_in_ = np.load('./DATA/test_input.npy')[test_ind,...]
test_out_ = np.load('./DATA/untrain_input.npy')

train_input_ = train_[:,1:]
val_input_ = val_[:,1:]
test_in_input_ = test_in_[:,1:]
test_out_input_ = test_out_[:,1:]

train_output_ = np.load('./DATA/train_output.npy')
val_output_ = np.load('./DATA/test_output.npy')[val_ind,...]
test_in_output_ = np.load('./DATA/test_output.npy')[test_ind,...]
test_out_output_ = np.load('./DATA/untrain_output.npy')

#%% LOAD SCALERS
with open('./DATA/IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
with open('./DATA/OUT_SCALER.pickle','rb') as f:
    OUT_SCALER = pickle.load(f)
#%%
SCALE = 'standard'
train_input = IN_SCALER[SCALE].transform(train_input_)
val_input = IN_SCALER[SCALE].transform(val_input_)
test_in_input = IN_SCALER[SCALE].transform(test_in_input_)
test_out_input = IN_SCALER[SCALE].transform(test_out_input_)
train_output = OUT_SCALER[SCALE].transform(train_output_.reshape(-1,1)).reshape(-1,360)
val_output = OUT_SCALER[SCALE].transform(val_output_.reshape(-1,1)).reshape(-1,360)
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
    # y_true_ = np.diff(y_true)
    # y_pred_ = np.diff(y_pred)
    y_true_ = y_true[...,1:]- y_true[...,:-1]
    y_pred_ = y_pred[...,1:]- y_pred[...,:-1]
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    diff_loss = tf.keras.losses.MSE(y_true_,y_pred_)
    loss = mse_loss + 0.1*diff_loss
    return loss
def diff_loss_ver2(y_true, y_pred):
    # y_true_ = np.diff(y_true)
    # y_pred_ = np.diff(y_pred)
    y_true_ = y_true[...,1:]- y_true[...,:-1]
    y_pred_ = y_pred[...,1:]- y_pred[...,:-1]
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    diff_loss = K.max(K.abs(y_true_-y_pred_),axis=1)
    # diff_loss = tf.keras.losses.MSE(y_true_,y_pred_)
    loss = mse_loss + 0.1*diff_loss
    return loss
def diff_loss_ver3(y_true, y_pred):
    # y_true_ = np.diff(y_true)
    # y_pred_ = np.diff(y_pred)
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

#%%
from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten, LeakyReLU,Conv1D, PReLU, Dropout, Concatenate
from tensorflow.keras.models import Model

def RNN_model(num_cell=128, dr_rates = 0.2, PE = True):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)
    if PE == True:
        pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
        inputs_extend = 0.5*pos_enc_tile+inputs_extend    
    layer_1 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(inputs_extend)       
    layer_2 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(layer_1)
    layer_3 = TimeDistributed(Dense(64,activation='relu'))(layer_2)
    outputs_ = TimeDistributed(Dense(1))(layer_3)
    outputs = Flatten()(outputs_)
    model= Model(inputs, outputs)
    return model

def RNN_model_ver2(num_cell=128, dr_rates = 0.2, PE = True, PE_type = 'add'):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)
    if PE == True:
        if PE_type=='add':
            pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
            inputs_extend = 0.5*pos_enc_tile+inputs_extend    
        else :
            pos_enc_tile = tf.tile(positional_encoding(360,PE_type), [num_rows, 1,1],name='pos_enc_tile')            
            inputs_extend = Concatenate()([inputs_extend,pos_enc_tile])

    layer_1 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(inputs_extend)       
    layer_2 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(layer_1)
    layer_3 = TimeDistributed(Dense(64,activation='relu'))(layer_2)
    outputs_ = TimeDistributed(Dense(1))(layer_3)
    outputs = Flatten()(outputs_)
    model= Model(inputs, outputs)
    return model

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

# def FNN_model():
#     inputs = Input(shape=(16))
#     layer_1 = Dense(128)(inputs)
#     layer_2 = PReLU()(layer_1)
#     layer_2 = Dropout(0.2)(layer_2)
#     layer_3 = Dense(256)(layer_2)
#     layer_4 = PReLU()(layer_3)
#     layer_4 = Dropout(0.2)(layer_4)
#     layer_5 = Dense(360)(layer_4)
#     layer_6 = PReLU()(layer_5)
#     layer_6 = Dropout(0.2)(layer_6)
#     layer_7 = Dense(360)(layer_6)
#     layer_8 = PReLU()(layer_7)
#     outputs = layer_8
#     model = Model(inputs,outputs)
#     return model
#%%
tf.random.set_seed(0)
vername = 'FR26'
base_model = RNN_model_v3(256,3,[128,64],0.2,8)
# base_model = RNN_model_ver2(256,0.2,PE=True, PE_type=8)
# base_model = FNN_model()
base_model.compile(optimizer='adam',
                # loss='mean_squared_error',
                loss = diff_loss,
                metrics=['mean_absolute_error','mean_squared_error']) 

ensure_dir('./MODEL/FR_RNN/{}'.format(vername))
path = './MODEL/FR_RNN/{}'.format(vername)+'/e{epoch:04d}.ckpt'
checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
            save_best_only = True,
            mode = 'auto',
            save_weights_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
hist=base_model.fit(train_input,train_output,
               validation_data=(val_input,val_output),
               callbacks=[checkpoint,reduce_lr,early_stopping],epochs=100,batch_size=256)
# with open('./MODEL/FR_RNN/{}/hist.pkl'.format(vername), 'wb') as f:
#         pickle.dump(hist.history,f)


#%%
# vername='FR23'
# latest = tf.train.latest_checkpoint('./MODEL/FR_RNN/{}'.format(vername))
# # base_model = RNN_model(256,0.2,PE=False)
# # base_model = RNN_model_ver2(256,0.2,PE=True, PE_type=8)
# base_model = RNN_model_v3(256,3,[128,64],0.2,'concat',8)
# base_model.load_weights(latest)


# with open('./MODEL/FR_RNN/{}/hist.pkl'.format(vername),'rb') as f:
#     history=pickle.load(f)
# #%%
# train_output_hat = base_model.predict(train_input,batch_size=128)
# train_output_hat_ = OUT_SCALER[SCALE].inverse_transform(train_output_hat.reshape(-1,1)).reshape(-1,360)
# val_output_hat = base_model.predict(val_input,batch_size=128)
# val_output_hat_ = OUT_SCALER[SCALE].inverse_transform(val_output_hat.reshape(-1,1)).reshape(-1,360)
# test_in_output_hat = base_model.predict(test_in_input,batch_size=128)
# test_in_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_in_output_hat.reshape(-1,1)).reshape(-1,360)
# test_out_output_hat = base_model.predict(test_out_input,batch_size=128)
# test_out_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_out_output_hat.reshape(-1,1)).reshape(-1,360)

# #%%
# np.round(np.mean(MAPE(train_output_, train_output_hat_)),3)
# np.round(np.mean(MAPE(val_output_, val_output_hat_)),3)
# np.round(np.mean(MAPE(test_in_output_, test_in_output_hat_)),3)
# np.round(np.mean(MAPE(test_out_output_, test_out_output_hat_)),3)

# np.round(np.mean(MSE(train_output_, train_output_hat_)),3)
# np.round(np.mean(MSE(val_output_, val_output_hat_)),3)
# np.round(np.mean(MSE(test_in_output_, test_in_output_hat_)),3)
# np.round(np.mean(MSE(test_out_output_, test_out_output_hat_)),3)


# %%
