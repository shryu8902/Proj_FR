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
with open('./MODEL/64qDiscretizer.pickle','rb') as f:
    discretizer = pickle.load(f)
#%%
SCALE = 'standard'
train_input = IN_SCALER[SCALE].transform(train_input_)
val_input = IN_SCALER[SCALE].transform(val_input_)z
test_in_input = IN_SCALER[SCALE].transform(test_in_input_)
test_out_input = IN_SCALER[SCALE].transform(test_out_input_)
train_output = OUT_SCALER[SCALE].transform(train_output_.reshape(-1,1)).reshape(-1,360)
val_output = OUT_SCALER[SCALE].transform(val_output_.reshape(-1,1)).reshape(-1,360)
test_in_output = OUT_SCALER[SCALE].transform(test_in_output_.reshape(-1,1)).reshape(-1,360)
test_out_output = OUT_SCALER[SCALE].transform(test_out_output_.reshape(-1,1)).reshape(-1,360)

#%%
diff_train_out=np.diff(train_output)
#%%
from sklearn.cluster import AgglomerativeClustering
agg_tr = AgglomerativeClustering(n_clusters=18,linkage='average')
agg_tr.fit(train_output)
label4 = agg_tr.labels_
labels4 = pd.DataFrame({'class':label4})
for i in list(set(label4)):
    fig, ax = plt.subplots()
    locs= np.where(label4==i)
    for j in locs[0]:
        ax.plot(train_output_[j,:])
    plt.title('hier_avg_cluster_{}'.format(i))
    plt.savefig('./Figs/hier_avg_{}_{}'.format(i,len(locs[0])))
    plt.clf()

#%%
from sklearn.cluster import MiniBatchKMeans
kmeans_tr = MiniBatchKMeans(n_clusters=10, random_state=0, batch_size=512)
kmeans_tr.fit(train_output)

km_train = kmeans_tr.predict(train_output)
km_val = kmeans_tr.predict(val_output)
km_test_in = kmeans_tr.predict(test_in_output)
km_test_out = kmeans_tr.predict(test_out_output)
with open('./DATA/kmeans.pkl','wb') as f:
    pickle.dump([km_train,km_val,km_test_in,km_test_out],f)
label2 = kmeans_tr.predict(test_out_output)
labels2 = pd.DataFrame({'class':label2})
for i in list(set(label2)):
    fig, ax = plt.subplots()
    locs= np.where(label2==i)
    for j in locs[0]:
        ax.plot(test_out_output_[j,:])
    plt.title('kmean_cluster_{}'.format(i))
    plt.savefig('./Figs/10KMCL_test_{}_{}'.format(i,len(locs[0])))
    plt.clf()


label2 = kmeans_tr.predict(train_output)
labels2 = pd.DataFrame({'class':label2})
for i in list(set(label2)):
    fig, ax = plt.subplots()
    locs= np.where(label2==i)
    for j in locs[0]:
        ax.plot(train_output_[j,:])
    plt.title('kmean_cluster_{}'.format(i))
    plt.savefig('./Figs/10KMCL_{}_{}'.format(i,len(locs[0])))
    plt.clf()
#%%
from sklearn.cluster import MiniBatchKMeans
diff_train_output_=np.diff(train_output_)
kmeans_tr_diff = MiniBatchKMeans(n_clusters=18, random_state=0, batch_size=512)
kmeans_tr_diff.fit(diff_train_output_)
label3 = kmeans_tr_diff.predict(diff_train_output_)
labels3 = pd.DataFrame({'class':label3})
for i in list(set(label3)):
    fig, ax = plt.subplots()
    locs= np.where(label3==i)
    for j in locs[0]:
        ax.plot(train_output_[j,:])
    plt.title('kmean_diff_cluster_{}'.format(i))
    plt.savefig('./Figs/DiffKMCL_{}_{}'.format(i,len(locs[0])))
    plt.clf()


#%%
from sklearn.cluster import DBSCAN
dbscan2 = DBSCAN(eps=5)
label2 = dbscan2.fit_predict(train_output)
labels = pd.DataFrame({'class':label})
labels['class'].value_counts()
len(set(label))

#%%
fig, ax = plt.subplots()
for i in list(set(label)):
    fig, ax = plt.subplots()
    locs= np.where(label==i)
    for j in locs[0]:
        ax.plot(train_output_[j,:])
    plt.title('cluster_'.format(i))
    plt.savefig('./Figs/CL_{}_{}'.format(i,len(locs[0])))
    plt.clf()

c_list = np.where(label==126)
for j in c_list[0]:
    plt.plot(train_output_[j,:])
# labels = pd.DataFrame({'cluster':label})


#%%
kmeans.predict(train_output)
k_centers = kmeans_tr.cluster_centers_
for i in range(40): 12,18, 20, 23,24, 30,32, 37
i=1
plt.plot(k_centers[i,:],label='cl_{}'.format(i))

c_list = np.where(kmeans_tr.predict(diff_train_out)==37)
len(c_list[0])
for j in c_list[0]:
    plt.plot(train_output_[j,:])
#%%
from scipy.cluster.hierarchy import dendrogram, linkage
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#%%

Z = linkage(train_output,'ward')
dn = dendrogram(Z)
plt.show()
# %%


Y = np.array(range(1400*350)).reshape((1400,350,1))

