#%%
import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('./Proj_FR')
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
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
#%%
from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(0.15, 0.25, 100)
wave = 1+0.5*signal.sawtooth(2 * np.pi * 5 * t)
meanwave = np.ones_like(t)*np.mean(wave)
delaywave = 1+0.5*signal.sawtooth(2 * np.pi * 5 * t-0.1*np.pi)
sinwave = 1+0.4*np.sin(2*np.pi*5*t-np.pi-np.sin(2*np.pi*5*t-np.pi-np.sin(2*np.pi*5*t-np.pi)/2)/2)
mse_mean=np.mean(np.square(wave-meanwave))
mse_delay = np.mean(np.square(wave-delaywave))
mse_sinwave = np.mean(np.square(wave-sinwave))

plt.plot(t, wave,label='original')
plt.plot(t,meanwave,label='mean mse:{}'.format(np.round(mse_mean,4)))
plt.plot(t, delaywave,label='delayed mse:{}'.format(np.round(mse_delay,4)))
plt.plot(t,sinwave,label='sinwave mse:{}'.format(np.round(mse_sinwave,4)))
plt.legend()
plt.grid()

#%%
mse_diff_mean=np.mean(np.square(np.diff(wave)-np.diff(meanwave)))
mse_diff_delay = np.mean(np.square(np.diff(wave)-np.diff(delaywave)))
mse_diff_sinwave = np.mean(np.square(np.diff(wave)-np.diff(sinwave)))
plt.plot(t[1:],np.diff(wave),label='original_diff')
plt.plot(t[1:],np.diff(meanwave),label='mean_wave_diff : {}'.format(np.round(mse_diff_mean,4)))
plt.plot(t[1:],np.diff(delaywave),label='delay_wave_diff : {}'.format(np.round(mse_diff_delay,4)))
plt.plot(t[1:],np.diff(sinwave),label='sin_wave_diff : {}'.format(np.round(mse_diff_sinwave,4)))

plt.grid()
plt.legend()

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

#%%
from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten, LeakyReLU,Conv1D, PReLU, Dropout
from tensorflow.keras.models import Model

def FNN_model(dr_rates=0.2):
    inputs = Input(shape=(16))
    layer_1 = Dense(256)(inputs)
    layer_2 = PReLU()(layer_1)
    layer_3 = Dense(256)(layer_2)
    layer_4 = PReLU()(layer_3)
    layer_5 = Dense(512)(layer_4)
    layer_6 = PReLU()(layer_5)
    layer_7 = Dropout(dr_rates)(layer_6)
    layer_8 = Dense(512)(layer_7)
    layer_9 = Dropout(dr_rates)(layer_8)
    layer_10 = PReLU()(layer_9)
    layer_11 = Dense(360)(layer_10)
    outputs = layer_11
    model = Model(inputs,outputs)
    return model
#%%
tf.random.set_seed(0)
vername = 'FRF4'
base_model = FNN_model(dr_rates=0.2)
# base_model = FNN_model()
base_model.compile(optimizer='adam',
                # loss='mean_squared_error',
                loss = diff_loss,
                metrics=['mean_absolute_error','mean_squared_error']) 

ensure_dir('./MODEL/FR_FNN/{}'.format(vername))
path = './MODEL/FR_FNN/{}'.format(vername)+'/e{epoch:04d}.ckpt'
checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
            save_best_only = True,
            mode = 'auto',
            save_weights_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
hist=base_model.fit(train_input,train_output,
               validation_data=(val_input,val_output),
               callbacks=[checkpoint,reduce_lr,early_stopping],epochs=100,batch_size=256)
with open('./MODEL/FR_FNN/{}/hist.pkl'.format(vername), 'wb') as f:
        pickle.dump(hist.history,f)


#%%
latest = tf.train.latest_checkpoint('./MODEL/FR_FNN/{}'.format(vername))
base_model = FNN_model(dr_rates=0.2)
base_model.load_weights(latest)


# with open('./MODEL/FR_RNN/FR6/hist.pkl','rb') as f:
#     history=pickle.load(f)
#%%
train_output_hat = base_model.predict(train_input,batch_size=256)
train_output_hat_ = OUT_SCALER[SCALE].inverse_transform(train_output_hat.reshape(-1,1)).reshape(-1,360)
val_output_hat = base_model.predict(val_input,batch_size=256)
val_output_hat_ = OUT_SCALER[SCALE].inverse_transform(val_output_hat.reshape(-1,1)).reshape(-1,360)
test_in_output_hat = base_model.predict(test_in_input,batch_size=256)
test_in_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_in_output_hat.reshape(-1,1)).reshape(-1,360)
test_out_output_hat = base_model.predict(test_out_input,batch_size=256)
test_out_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_out_output_hat.reshape(-1,1)).reshape(-1,360)

#%%
np.round(np.mean(MAPE(train_output_, train_output_hat_)),3)
np.round(np.mean(MAPE(val_output_, val_output_hat_)),3)
np.round(np.mean(MAPE(test_in_output_, test_in_output_hat_)),3)
np.round(np.mean(MAPE(test_out_output_, test_out_output_hat_)),3)

np.round(np.mean(MSE(train_output_, train_output_hat_)),3)
np.round(np.mean(MSE(val_output_, val_output_hat_)),3)
np.round(np.mean(MSE(test_in_output_, test_in_output_hat_)),3)
np.round(np.mean(MSE(test_out_output_, test_out_output_hat_)),3)


# %%
