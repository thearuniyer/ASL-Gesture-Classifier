#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import os
import scipy.ndimage as im
from scipy.signal import medfilt
from scipy.signal import savgol_filter


# In[3]:


def modify_action_files(folder):
    all_files=os.listdir(folder)
    #print(all_files)
    for csv in all_files:
        path=os.path.join(folder+'\\'+csv)
        df=pd.read_csv(path)
        df=df.drop(['Frames#','score_overall','nose_score','leftEye_score','rightEye_score','leftEar_score','rightEar_score','leftShoulder_score','rightShoulder_score','leftElbow_score','rightElbow_score','leftWrist_score','rightWrist_score','leftHip_score','rightHip_score','leftKnee_score','rightKnee_score','leftAnkle_score','rightAnkle_score'],axis=1)
        df.to_csv(path,index=False)


# In[ ]:


def drop_cols_df(df):
    df=df.drop(['Frames#','score_overall','nose_score','leftEye_score','rightEye_score','leftEar_score','rightEar_score','leftShoulder_score','rightShoulder_score','leftElbow_score','rightElbow_score','leftWrist_score','rightWrist_score','leftHip_score','rightHip_score','leftKnee_score','rightKnee_score','leftAnkle_score','rightAnkle_score'],axis=1)
    return df


# In[4]:


def remove_org_cols(folder):
    all_files=os.listdir(folder)
    #print(all_files)
    for csv in all_files:
        path=os.path.join(folder+'\\'+csv)
        df=pd.read_csv(path)
        df=df.drop(['nose_x', 'nose_y', 'leftEye_x', 'leftEye_y', 'rightEye_x',
       'rightEye_y', 'leftEar_x', 'leftEar_y', 'rightEar_x', 'rightEar_y',
       'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_x',
       'rightShoulder_y', 'leftElbow_x', 'leftElbow_y', 'rightElbow_x',
       'rightElbow_y', 'leftWrist_x', 'leftWrist_y', 'rightWrist_x',
       'rightWrist_y', 'leftHip_x', 'leftHip_y', 'rightHip_x', 'rightHip_y',
       'leftKnee_x', 'leftKnee_y', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_x',
       'leftAnkle_y', 'rightAnkle_x', 'rightAnkle_y'],axis=1)
        df.to_csv(path,index=False)


# In[8]:


def remove_org_cols_df(df):
    df=df.drop(['nose_x', 'nose_y', 'leftEye_x', 'leftEye_y', 'rightEye_x',
       'rightEye_y', 'leftEar_x', 'leftEar_y', 'rightEar_x', 'rightEar_y',
       'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_x',
       'rightShoulder_y', 'leftElbow_x', 'leftElbow_y', 'rightElbow_x',
       'rightElbow_y', 'leftWrist_x', 'leftWrist_y', 'rightWrist_x',
       'rightWrist_y', 'leftHip_x', 'leftHip_y', 'rightHip_x', 'rightHip_y',
       'leftKnee_x', 'leftKnee_y', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_x',
       'leftAnkle_y', 'rightAnkle_x', 'rightAnkle_y'],axis=1)
    return df


# In[5]:


def get_rel_score(folder):
    all_files=os.listdir(folder)
    #print(all_files)
    for csv in all_files:
        path=os.path.join(folder+'\\'+csv)
        df=pd.read_csv(path)
        #print(df.shape)
        nose_x_vec=df['nose_x'].values
        nose_y_vec=df['nose_y'].values
        j=2
        i=0
        count=2
        name='rel_'
        while(count<34):
            x2_vec=df.iloc[:,j].values
            y2_vec=df.iloc[:,j+1].values
            #print(j)
            (new_col_1,new_col_2)=get_dist(nose_x_vec,nose_y_vec,x2_vec,y2_vec)
            new_name_1=name+str(i)+'x'
            new_name_2=name+str(i)+'y'
            df[new_name_1]=new_col_1
            df[new_name_2]=new_col_2
            i+=1
            j+=2
            count+=2
        df.to_csv(path,index=False)

def get_dist(nose_x_vec,nose_y_vec,x2_vec,y2_vec):
    list_1=np.absolute(nose_x_vec-x2_vec)
    list_2=np.absolute(nose_y_vec-y2_vec)
    #print(list_1)
    #print("\n")
    #print(list_2)
    return (list_1,list_2)


# In[7]:


def get_rel_score_df(df):
    nose_x_vec=df['nose_x'].values
    nose_y_vec=df['nose_y'].values
    j=2
    i=0
    count=2
    name='rel_'
    while(count<34):
        x2_vec=df.iloc[:,j].values
        y2_vec=df.iloc[:,j+1].values
        #print(j)
        (new_col_1,new_col_2)=get_dist(nose_x_vec,nose_y_vec,x2_vec,y2_vec)
        new_name_1=name+str(i)+'x'
        new_name_2=name+str(i)+'y'
        df[new_name_1]=new_col_1
        df[new_name_2]=new_col_2
        i+=1
        j+=2
        count+=2
    return df


# In[6]:


# folder=r'C:\Users\mahat\Documents\CSV_data_Tuesday\CSV\buy'
# modify_action_files(folder)
# get_rel_score(folder)
# remove_org_cols(folder)


# In[ ]:
