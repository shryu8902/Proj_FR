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
import streamlit as st

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

st.set_page_config(page_title='FR_Result',page_icon='🚨', layout="wide")

#%%load data
val_ind,test_ind = train_test_split(range(25988),test_size=18000, random_state=0)
train_ = np.load('/home/ashryu/Proj_FR/DATA/train_input.npy')
val_ = np.load('/home/ashryu/Proj_FR/DATA/test_input.npy')[val_ind,...]
test_in_ = np.load('/home/ashryu/Proj_FR/DATA/test_input.npy')[test_ind,...]
test_out_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_input.npy')

train_input_ = train_[:,1:]
val_input_ = val_[:,1:]
test_in_input_ = test_in_[:,1:]
test_out_input_ = test_out_[:,1:]

train_output_ = np.load('/home/ashryu/Proj_FR/DATA/train_output.npy')
val_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[val_ind,...]
test_in_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[test_ind,...]
test_out_output_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_output.npy')
#%% LOAD SCALERS
with open('/home/ashryu/Proj_FR/DATA/IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
with open('/home/ashryu/Proj_FR/DATA/OUT_SCALER.pickle','rb') as f:
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
#%%
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
result = pd.DataFrame(columns =['version','seed','MAPE_val','MAPE_in','MAPE_out','MSE_val','MSE_in','MSE_out','diff_MSE_in','diff_MSE_out'])
for version in tqdm.tqdm(range(11)):
    ver=version+1
    for seed in range(5):
        vername = 'RNN-{}-S{}'.format(ver,seed)
        if ver==1:
            model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None)
        elif ver==2:
            model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None)
        elif ver ==3:
            model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='add')
        elif ver ==4:
            model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='concat',PE_d=8)
        elif ver ==5:
            model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='concat',PE_d=8)
        elif ver == 10:
            model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None)    
        elif ver == 11 :
            model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='concat',PE_d=8)
        else:
            continue
        latest = tf.train.latest_checkpoint('./MODEL/{}'.format(vername))
        model.load_weights(latest)

        # train_output_hat = base_model.predict(train_input,batch_size=128)
        # train_output_hat_ = OUT_SCALER[SCALE].inverse_transform(train_output_hat.reshape(-1,1)).reshape(-1,360)
        val_output_hat = model.predict(val_input,batch_size=128)
        val_output_hat_ = OUT_SCALER[SCALE].inverse_transform(val_output_hat.reshape(-1,1)).reshape(-1,360)        
        test_in_output_hat = model.predict(test_in_input,batch_size=128)
        test_in_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_in_output_hat.reshape(-1,1)).reshape(-1,360)
        test_out_output_hat = model.predict(test_out_input,batch_size=128)
        test_out_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_out_output_hat.reshape(-1,1)).reshape(-1,360)

        # np.round(np.mean(MAPE(train_output_, train_output_hat_)),3)
        # np.round(np.mean(MAPE(val_output_, val_output_hat_)),3)
        mape_val = np.round(np.mean(MAPE(val_output_, val_output_hat_)),3)
        mape_in = np.round(np.mean(MAPE(test_in_output_, test_in_output_hat_)),3)
        mape_out = np.round(np.mean(MAPE(test_out_output_, test_out_output_hat_)),3)       
        mse_val = np.round(np.mean(MSE(val_output_, val_output_hat_)),3)
        mse_in = np.round(np.mean(MSE(test_in_output_, test_in_output_hat_)),3)
        mse_out = np.round(np.mean(MSE(test_out_output_, test_out_output_hat_)),3)
        diff_mse_in = np.round(np.mean(MSE(np.diff(test_in_output_), np.diff(test_in_output_hat_))),3)
        diff_mse_out = np.round(np.mean(MSE(np.diff(test_out_output_), np.diff(test_out_output_hat_))),3)

        result=result.append({'version':ver,'seed':seed,'MAPE_val':mape_val,'MAPE_in':mape_in,'MAPE_out':mape_out,
                            'MSE_val':mse_val,'MSE_in':mse_in,'MSE_out':mse_out,
                            'diff_MSE_in':diff_mse_in,'diff_MSE_out':diff_mse_out},ignore_index=True)
#%%
FR_result={}
# result = pd.DataFrame(columns =['version','seed','MAPE_val','MAPE_in','MAPE_out','MSE_val','MSE_in','MSE_out','diff_MSE_in','diff_MSE_out'])
for version in tqdm.tqdm(range(5)):
    ver=version+1
    if ver==1:
        model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None)
        seed=3
    elif ver==2:
        model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE=None)
        seed=4
    elif ver ==3:
        model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='add')
        seed=4
    elif ver ==4:
        model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='concat',PE_d=8)
        seed=0
    elif ver ==5:
        model = RNN_model_v3(256,2,[128,64],dr_rates=0.2, PE='concat',PE_d=8)
        seed=2
    vername = 'RNN-{}-S{}'.format(ver,seed)
    latest = tf.train.latest_checkpoint('./MODEL/{}'.format(vername))
    model.load_weights(latest)

    test_in_output_hat = model.predict(test_in_input,batch_size=128)
    test_in_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_in_output_hat.reshape(-1,1)).reshape(-1,360)
    test_out_output_hat = model.predict(test_out_input,batch_size=128)
    test_out_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_out_output_hat.reshape(-1,1)).reshape(-1,360)
    FR_result['RNN-{}'.format(ver)]={}
    FR_result['RNN-{}'.format(ver)]['in']=test_in_output_hat_
    FR_result['RNN-{}'.format(ver)]['out']=test_out_output_hat_

#%%
# untrain : SGTR: 273(0),274(1),141825(17998),141826(17999)141790
index = 81000
df_test = pd.DataFrame(test_out_)
# iloc = df_test.loc[df_test[0]==index].index[0]
# if df_test.iloc[iloc][16] == 0:
#     ttype = 'SGTR'
# else :
#     ttype = 'MSLB'
iloc=14000    

plt.plot((test_out_output_[iloc,:]),label='True value')
plt.plot((FR_result['RNN-11']['out'][iloc,:]),label='Basic')
# plt.plot((FR_result['RNN-2']['out'][iloc,:]),label='Diff_loss')
# plt.plot((FR_result['RNN-3']['out'][iloc,:]),label='PE(add)')
# plt.plot((FR_result['RNN-4']['out'][iloc,:]),label='PE(concat)')
# plt.plot((FR_result['RNN-5']['out'][iloc,:]),label='PE(concat) & Diff_loss')

# plt.plot(np.diff(test_out_output_[iloc,:]),label='True value')
# plt.plot(np.diff(FR_result['RNN-1']['out'][iloc,:]),label='Basic')
# plt.plot(np.diff(FR_result['RNN-2']['out'][iloc,:]),label='Diff_loss')
# plt.plot(np.diff(FR_result['RNN-3']['out'][iloc,:]),label='PE(add)')
# plt.plot(np.diff(FR_result['RNN-4']['out'][iloc,:]),label='PE(concat)')
# plt.plot(np.diff(FR_result['RNN-5']['out'][iloc,:]),label='PE(concat) & Diff_loss')
# plt.plot(test_output_hat_hat_[iloc,:],label='Fine')
plt.tight_layout()
plt.legend()
plt.grid()
plt.ylabel('Primary Pressure ($kg/cm^2\cdot a$)')
plt.xlabel('Time (10 seconds)')
plt.title('Test set, Index : {}, Type: {}'.format(str(index),'ttype'))
# plt.savefig('./Figs/1-1.png',dpi=300)

#%%
i=15000
fig, ax1 = plt.subplots()
plt.plot(test_in_output_[i]/100)
ax2 = ax1.twinx()
plt.plot(np.diff(test_in_output_[i]))
plt.plot(np.diff(test_in_output_hat_[i]))



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
