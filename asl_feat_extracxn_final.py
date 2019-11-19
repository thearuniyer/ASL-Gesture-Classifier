#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import os
import scipy
import scipy.ndimage as im
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import math
import pywt
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from scipy.fftpack import fft
from sklearn.preprocessing import scale
import tsfresh
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[136]:


def feat_ext_folder(folder):
    all_files=os.listdir(folder)
    #print(all_files)
    feat_mat=[]
    for csv in all_files:
        path=os.path.join(folder+'\\'+csv)
        df=pd.read_csv(path)
        df=df.iloc[:,8:24]
        feat_list=feat_ext(df)
        feat_mat.append(feat_list)
    return feat_mat


# In[137]:


def feat_ext(df):
    feat=[]
    feat+=[abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(df[column],[{"coeff": 1, "attr": "real"}]))[0][1]) for column in df]
    feat+=[abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(df[column],[{"coeff": 2, "attr": "real"}]))[0][1]) for column in df]
    feat+=[abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(df[column],[{"coeff": 3, "attr": "real"}]))[0][1]) for column in df]
    feat+=[abs(list(tsfresh.feature_extraction.feature_calculators.spkt_welch_density(df[column],[{"coeff": 1}]))[0][1]) for column in df]
    feat+=[abs(list(tsfresh.feature_extraction.feature_calculators.spkt_welch_density(df[column],[{"coeff": 2}]))[0][1]) for column in df]
    feat+=[abs(list(tsfresh.feature_extraction.feature_calculators.spkt_welch_density(df[column],[{"coeff": 3}]))[0][1]) for column in df]
    feat+=[tsfresh.feature_extraction.feature_calculators.kurtosis(df[column]) for column in df]
    feat+=[tsfresh.feature_extraction.feature_calculators.sample_entropy(df[column]) for column in df]
    feat+=[tsfresh.feature_extraction.feature_calculators.skewness(df[column]) for column in df]
    feat+=[tsfresh.feature_extraction.feature_calculators.autocorrelation(df[column],2)for column in df]
    feat+=[tsfresh.feature_extraction.feature_calculators.number_peaks(df[column],2) for column in df]
    feat+=[tsfresh.feature_extraction.feature_calculators.median(df[column]) for column in df]
    return feat


# In[138]:


def list_to_arr(mat_list):
 dum_np=np.array(mat_list)
 return dum_np


# In[139]:


def create_Y_matrix(num1,num2):
    Y_mat=[num1]*num2
    Y_mat=np.reshape(Y_mat,(num2,1))
    return Y_mat


# In[140]:


# folder=r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\buy'
# buy_mat=list_to_arr(feat_ext_folder(folder))
# #print(len(buy_mat))
# folder=r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\communicate'
# com_mat=list_to_arr(feat_ext_folder(folder))
# folder=r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\fun'
# fun_mat=list_to_arr(feat_ext_folder(folder))
# folder=r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\hope'
# hope_mat=list_to_arr(feat_ext_folder(folder))
# folder=r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\mother'
# mom_mat=list_to_arr(feat_ext_folder(folder))
# folder=r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\really'
# really_mat=list_to_arr(feat_ext_folder(folder))


# In[141]:


# X_matrix=np.vstack((buy_mat,com_mat,fun_mat,hope_mat,mom_mat,really_mat))
# print(X_matrix.shape)
# buy_y_mat=create_Y_matrix(0,buy_mat.shape[0])
# com_y_mat=create_Y_matrix(1,com_mat.shape[0])
# fun_y_mat=create_Y_matrix(2,fun_mat.shape[0])
# hope_y_mat=create_Y_matrix(3,hope_mat.shape[0])
# mom_y_mat=create_Y_matrix(4,mom_mat.shape[0])
# really_y_mat=create_Y_matrix(5,really_mat.shape[0])
# Y_matrix=np.vstack((buy_y_mat,com_y_mat,fun_y_mat,hope_y_mat,mom_y_mat,really_y_mat))
# print(Y_matrix.shape)
# dataset=np.hstack((X_matrix,Y_matrix))
# print(dataset.shape)


# In[144]:


# '''This is just to set a learning model benchmark '''
# X_matrix, Y_matrix = shuffle(X_matrix, Y_matrix)
# X_train, X_test, Y_train, Y_test = train_test_split(X_matrix, Y_matrix, test_size=0.2, random_state=42)
# knn = KNeighborsClassifier(n_neighbors=3)
# ## Fit the model on the training data.
# knn.fit(X_train, Y_train)
# ## See how the model performs on the test data.
# knn.score(X_test, Y_test)


# In[145]:


# np.savetxt(r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\data_X.txt',X_matrix)
# np.savetxt(r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\data_Y.txt',Y_matrix)
