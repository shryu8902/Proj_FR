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

#%%
train_input_ = np.load('F:/gen_data/train_input.npy')[:,1:]
val_input_ = np.load('F:/gen_data/test_input.npy')[:,1:]
test_input_ = np.load('F:/gen_data/untrain_input.npy')[:,1:]
train_output_ = np.load('F:/gen_data/train_output.npy')
val_output_ = np.load('F:/gen_data/test_output.npy')
test_output_ = np.load('F:/gen_data/untrain_output.npy')
train_ = np.load('F:/gen_data/train_input.npy')
val_ = np.load('F:/gen_data/test_input.npy')
test_ = np.load('F:/gen_data/untrain_input.npy')
#%%
## SAVE SCALERS
# OUT_SCALER = {'standard':StandardScaler(),'minmax':MinMaxScaler()}
# IN_SCALER = {'standard':StandardScaler(),'minmax':MinMaxScaler()}
# for dat in [train_input,val_input,test_input]:
#     IN_SCALER['standard'].partial_fit(dat)
#     IN_SCALER['minmax'].partial_fit(dat)

# for dat in [train_output,val_output,test_output]:
#     OUT_SCALER['standard'].partial_fit(dat.reshape(-1,1))
#     OUT_SCALER['minmax'].partial_fit(dat.reshape(-1,1))
#  with open('./IN_SCALER.pickle','wb') as f:
#      pickle.dump(IN_SCALER,f)
#  with open('./OUT_SCALER.pickle','wb') as f:
#      pickle.dump(OUT_SCALER,f)

## LOAD SCALERS
with open('./IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
with open('./OUT_SCALER.pickle','rb') as f:
    OUT_SCALER = pickle.load(f)
#%%
with open('F:/ie_diagnosis/SCALERS.pickle','rb') as f:
    SCALERS=pickle.load(f)
SCALE = 'standard'
train_input = IN_SCALER[SCALE].transform(train_input_)
val_input = IN_SCALER[SCALE].transform(val_input_)
test_input = IN_SCALER[SCALE].transform(test_input_)
train_output = OUT_SCALER[SCALE].transform(train_output_.reshape(-1,1)).reshape(-1,360)
val_output = OUT_SCALER[SCALE].transform(val_output_.reshape(-1,1)).reshape(-1,360)
test_output = OUT_SCALER[SCALE].transform(test_output_.reshape(-1,1)).reshape(-1,360)
# del(train_input_,test_input_,val_input_,train_output_,test_output_,val_output_)

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
from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten
from tensorflow.keras.models import Model
def RNN_model(num_cell=128, dr_rates = 0.3):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)
    input_with_PE = pos_enc_tile+inputs_extend
    layer_1 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(input_with_PE)   
    layer_2 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(layer_1)
    layer_3 = TimeDistributed(Dense(64,activation='selu'))(layer_2)
    outputs_ = TimeDistributed(Dense(1))(layer_3)
    outputs = Flatten()(outputs_)
    model= Model(inputs, outputs)
    return model
#%%
def RNN_model(num_cell=128, dr_rates = 0.3):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)
    input_with_PE = pos_enc_tile+inputs_extend
    layer_1 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(input_with_PE)   
    layer_2 = TimeDistributed(Dense(64))(layer_1)
    layer_3 = TimeDistributed(LeakyReLU())(layer_2)
    layer_4 = BatchNormalization()(layer_3)
    outputs_ = TimeDistributed(Dense(1))(layer_4)
    outputs = Flatten()(outputs_)
    model= Model(inputs, outputs)
    return model
#%%
from tensorflow.keras.layers import Input, RepeatVector,Conv1D, BatchNormalization, LeakyReLU, Conv2DTranspose,Flatten

def CNN_model(num_cell=128, dr_Rates = 0.3):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)
    input_with_PE = pos_enc_tile+inputs_extend

    layer_1 = Conv1D(num_cell,3,strides=3)(input_with_PE)
    layer_1_1 = BatchNormalization()(layer_1)
    layer_1_2 = LeakyReLU()(layer_1_1) 
    layer_2 = Conv1D(num_cell,3,strides=3)(layer_1_2)
    layer_2_1 = BatchNormalization()(layer_2)
    layer_2_2 = LeakyReLU()(layer_2_1) 
    layer_3 = Conv1D(num_cell,2,strides=2)(layer_2_2)
    layer_3_1 = BatchNormalization()(layer_3)
    layer_3_2 = LeakyReLU()(layer_3_1) 
    layer_3_3 = tf.expand_dims(layer_3_2,1)
    layer_4 = Conv2DTranspose(num_cell,(1,2),(1,2))(layer_3_3)
    layer_4_1 = BatchNormalization()(layer_4)
    layer_4_2 = LeakyReLU()(layer_4_1) 
    layer_5 = Conv2DTranspose(num_cell,(1,3),(1,3))(layer_4_2)
    layer_5_1 = BatchNormalization()(layer_5)
    layer_5_2 = LeakyReLU()(layer_5_1) 
    layer_6 = Conv2DTranspose(1,(1,3),(1,3))(layer_5_2)
    layer_6_1 = BatchNormalization()(layer_6)
    layer_6_2 = LeakyReLU()(layer_6_1) 
    outputs = Flatten()(layer_6_2)
    model= Model(inputs, outputs)
    return model

#%% ver6 (256)
vername = 'ver8'
base_model = RNN_model(512,0.2)
base_model.compile(optimizer='adam',
                # loss='mean_squared_error',
                loss = diff_loss,
                metrics=['mean_absolute_error','mean_squared_error']) 

ensure_dir('./Model/FR_RNN/{}'.format(vername))
path = './Model/FR_RNN/{}'.format(vername)+'/e{epoch:04d}.ckpt'
checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
            save_best_only = True,
            mode = 'auto',
            save_weights_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
hist=base_model.fit(train_input,train_output,
               validation_data=(val_input,val_output),
               callbacks=[checkpoint,reduce_lr,early_stopping],epochs=100,batch_size=256)
with open('./Model/FR_RNN/{}/hist.pkl'.format(vername), 'wb') as f:
        pickle.dump(hist.history,f)

#%%
vername = 'ver1'
base_model = CNN_model(128,0.3)
base_model.compile(optimizer='adam',
                # loss='mean_squared_error',
                loss = diff_loss,
                metrics=['mean_absolute_error','mean_squared_error']) 

ensure_dir('./Model/FR_CNN/{}'.format(vername))
path = './Model/FR_CNN/{}'.format(vername)+'/e{epoch:04d}.ckpt'
checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
            save_best_only = True,
            mode = 'auto',
            save_weights_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
hist=base_model.fit(train_input,train_output,
               validation_data=(val_input,val_output),
               callbacks=[checkpoint,reduce_lr,early_stopping],epochs=100,batch_size=256)
with open('./Model/FR_CNN/{}/hist.pkl'.format(vername), 'wb') as f:
        pickle.dump(hist.history,f)

#%%
vername = 'ver5'
base_model = RNN_model(128,0.3)
base_model.compile(optimizer='adam',
                # loss='mean_squared_error',
                loss = diff_loss,
                metrics=['mean_absolute_error','mean_squared_error']) 

path = './Model/FR_RNN/{}'.format(vername)
latest = tf.train.latest_checkpoint(path)
base_model.load_weights(latest)


#%%
test_output_hat = base_model.predict(test_input)
test_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_output_hat.reshape(-1,1)).reshape(-1,360)
val_output_hat = base_model.predict(val_input)
val_output_hat_ = OUT_SCALER[SCALE].inverse_transform(val_output_hat.reshape(-1,1)).reshape(-1,360)

def MAPE(y_true,y_pred):
    ape = abs(y_true-y_pred)/y_true*100
    mape = np.mean(ape,axis=-1)
    return mape
def MSE(y_true,y_pred):
    se = (y_true-y_pred)**2
    mse = np.mean(se,axis=-1)
    return mse

np.mean(MAPE(test_output_, test_output_hat_))
np.mean(MSE(test_output_, test_output_hat_))
np.mean(MAPE(val_output_, val_output_hat_))
np.mean(MSE(val_output_, val_output_hat_))

#%%
# untrain : SGTR: 273(0),274(1),141825(17998),141826(17999)141790
index = 141749
df_test = pd.DataFrame(test_out_)
iloc = df_test.loc[df_test[0]==index].index[0]
if df_test.iloc[iloc][16] == 0:
    ttype = 'SGTR'
else :
    ttype = 'MSLB'
plt.plot(test_out_output_[iloc,:],label='True')
plt.plot(test_out_output_hat_[iloc,:],label='Fast running')
# plt.plot(test_output_hat_hat_[iloc,:],label='Fine')

plt.legend()
plt.grid()

plt.title('Index : {}, Type: {}'.format(str(index),ttype))
# plt.savefig('./test_diff_{}.png'.format(str(index)))
#%%
# val : SGTR : 26848(0),18902(1), MSLB 113865(7), 139096(8)
index = 113865
df_test = pd.DataFrame(val_)
iloc = df_test.loc[df_test[0]==index].index[0]
if df_test.iloc[iloc][16] == 0:
    ttype = 'SGTR'
else :
    ttype = 'MSLB'
plt.plot(val_output_[iloc,:],label='True')
plt.plot(val_output_hat_[iloc,:],label='Fast running')
plt.legend()
plt.grid()
plt.title('Index : {}, Type: {}'.format(str(index),ttype))
# plt.savefig('./val_{}.png'.format(str(index)))

#%%

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#%% 
        history = model.fit(X_train, Y_train.reshape(-1,520,1), 
                        validation_data=(X_val,Y_val),
                        batch_size = 512, epochs=100,
                        callbacks=[early_stopping,model_save]))
 
#%%

