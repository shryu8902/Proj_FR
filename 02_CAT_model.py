#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def softlabel(input_data):
    new_input = np.zeros_like(input_data)
    for i,matrix in enumerate(input_data):
        for j,vector in enumerate(matrix):
            k = np.convolve(vector,[0.05,0.1,0.7,0.1,0.05],'same')
            if np.sum(k)!=1:
                k=k/np.sum(k)
            new_input[i,j,...]=k
    return new_input

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#%%
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
#%%
train_ = np.load('./DATA/train_input.npy')
val_ = np.load('./DATA/test_input.npy')
test_ = np.load('./DATA/untrain_input.npy')

train_input_ = train_[:,1:]
val_input_ = val_[:,1:]
test_input_ = test_[:,1:]
train_output_ = np.load('./DATA/train_output.npy')
val_output_ = np.load('./DATA/test_output.npy')
test_output_ = np.load('./DATA/untrain_output.npy')
#%%
from sklearn.preprocessing import KBinsDiscretizer
NBIN=128

discretizer = KBinsDiscretizer(n_bins=NBIN, encode='onehot-dense', strategy='quantile')
discretizer.fit(train_output_.reshape(-1,1))
#%%
train_output_prob = discretizer.transform(train_output_.reshape(-1,1)).reshape(-1,360,NBIN)
val_output_prob = discretizer.transform(val_output_.reshape(-1,1)).reshape(-1,360,NBIN)
test_output_prob = discretizer.transform(test_output_.reshape(-1,1)).reshape(-1,360,NBIN)

train_out_prob = softlabel(train_output_prob)
val_out_prob = softlabel(val_output_prob)

#%% LOAD SCALERS
with open('./DATA/IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
#%%
SCALE = 'standard'
train_input = IN_SCALER[SCALE].transform(train_input_)
val_input = IN_SCALER[SCALE].transform(val_input_)
test_input = IN_SCALER[SCALE].transform(test_input_)

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
    loss = mse_loss + diff_loss
    return loss

#%%
from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten, LeakyReLU,Conv1D
from tensorflow.keras.models import Model
def RNN_model(num_cell=128, dr_rates = 0.3,n_bins=128):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)
    input_with_PE = pos_enc_tile+inputs_extend
    layer_1 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(input_with_PE)   
    layer_2 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(layer_1)
    layer_3 = TimeDistributed(Dense(192))(layer_2)
    layer_3 = TimeDistributed(LeakyReLU())(layer_3)
    outputs_ = TimeDistributed(Dense(n_bins,activation='softmax'))(layer_3)
    # fine_layer_1_1 = Conv1D(64,3,1,padding='same')(outputs_)
    # fine_layer_1_2 = LeakyReLU()(fine_layer_1_1)
    # fine_layer_1_3 = Conv1D(1,3,1,padding='same')(fine_layer_1_2) 
    # fine_layer_1 = outputs_ + fine_layer_1_3
    # fine_layer_2_1 = Conv1D(64,3,1,padding='same')(fine_layer_1)
    # fine_layer_2_2 = LeakyReLU()(fine_layer_2_1)
    # fine_layer_2_3 = Conv1D(1,3,1,padding='same')(fine_layer_2_2) 
    # fine_layer_2 = fine_layer_1+fine_layer_2_3
    # fine_outputs=Flatten()(fine_layer_2)
    model= Model(inputs, outputs_)
    return model
#%%
tf.random.set_seed(0)
#ver3 : label smoothing dense 64
#ver4 : label smoothing with leaky relu and 192 dense

vername = 'ver5'
base_model = RNN_model(256,0.2)
base_model.compile(optimizer='adam',
                # loss='mean_squared_error',
                loss = 'categorical_crossentropy',
                metrics=['accuracy','mean_absolute_error']) 

ensure_dir('./MODEL/FR_CAT_RNN/{}'.format(vername))
path = './MODEL/FR_CAT_RNN/{}'.format(vername)+'/e{epoch:04d}.ckpt'
checkpoint = ModelCheckpoint(path, monitor = 'val_accuracy',verbose = 1,
            save_best_only = True,
            mode = 'auto',
            save_weights_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
hist=base_model.fit(train_input,train_out_prob,
               validation_data=(val_input,val_out_prob),
               callbacks=[checkpoint,reduce_lr,early_stopping],epochs=100,batch_size=128)
with open('./MODEL/FR_CAT_RNN/{}/hist.pkl'.format(vername), 'wb') as f:
        pickle.dump(hist.history,f)

#%%
#%%
latest = tf.train.latest_checkpoint('./MODEL/FR_CAT_RNN/{}'.format(vername))
base_model = RNN_model(512,0.2)
base_model.load_weights(latest)
#%%
test_output_hat_prob = base_model.predict(test_input,batch_size=256)
test_output_hat_prob = (test_output_hat_prob == test_output_hat_prob.max(axis=2, keepdims=1)).astype(float)
test_output_hat_ = discretizer.inverse_transform(test_output_hat_prob.reshape(-1,NBIN)).reshape(-1,360)
val_output_hat_prob = base_model.predict(val_input,batch_size=256)
val_output_hat_prob = (val_output_hat_prob == val_output_hat_prob.max(axis=2, keepdims=1)).astype(float)
val_output_hat_ = discretizer.inverse_transform(val_output_hat_prob.reshape(-1,NBIN)).reshape(-1,360)

#%%
np.mean(MAPE(test_output_, test_output_hat_))
np.mean(MSE(test_output_, test_output_hat_))
np.mean(MAPE(val_output_, val_output_hat_))
np.mean(MSE(val_output_, val_output_hat_))
