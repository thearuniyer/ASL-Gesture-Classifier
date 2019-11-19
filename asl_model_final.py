#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
#from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import pathlib


# In[2]:


def get_model_performance(model,X_train,Y_train,X_test,Y_test):
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model.fit(X_train,Y_train)
    #print(cross_val_score(model, X_train, Y_train, cv=5))
    print(model.score(X_test, Y_test))


# In[3]:


def get_grid_search_params(model,param_grid,X,Y):
    model_gscv = GridSearchCV(model, param_grid, cv=5)
    model_gscv.fit(X, Y)
    return model_gscv.best_params_   


# In[5]:


X_data=np.loadtxt(r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\data_X.txt')
Y_data=np.loadtxt(r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\data_Y.txt')
clf = DecisionTreeClassifier()
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
sm = SMOTE(random_state = 2) 
X_train, Y_train = sm.fit_sample(X_train, Y_train) 
param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15),'random_state':[0,1,2,3,4,5]}
#get_grid_search_params(clf,param_grid,X_train,Y_train)
#get_model_performance(clf,X_data,Y_data)


# In[6]:


clf=DecisionTreeClassifier(criterion='entropy',max_depth=7,random_state=2)
get_model_performance(clf,X_train,Y_train,X_test,Y_test)


# In[72]:


Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
clf=SVC(kernel='rbf')
get_grid_search_params(clf,param_grid,X_train,Y_train)


# In[73]:


clf=SVC(kernel='rbf',C=0.001,gamma=0.001)
get_model_performance(clf,X_train,Y_train,X_test,Y_test)


# In[254]:


param_grid = { 
    'n_estimators': [50, 100, 150, 200],
    'max_features': ['auto'],
    'max_depth' : [4,5,6,7,8,9,10,11,13],
    'criterion' :['gini', 'entropy'], 'random_state':[0,1,2,3,4,5]
}
clf=RandomForestClassifier()
get_grid_search_params(clf,param_grid,X_train,Y_train)


# In[7]:


clf=RandomForestClassifier(random_state=3,criterion='entropy',max_depth=11,max_features='auto',n_estimators=200)
get_model_performance(clf,X_train,Y_train,X_test,Y_test)


# In[218]:


param_grid={'n_neighbors':[1,2,3,4,5],'weights':['uniform','distance'],'metric':['euclidean','manhattan']}
clf = KNeighborsClassifier()
get_grid_search_params(clf,param_grid,X_train,Y_train)


# In[8]:


clf=KNeighborsClassifier(n_neighbors=1,metric='manhattan',weights='uniform')
get_model_performance(clf,X_train,Y_train,X_test,Y_test)


# In[9]:


clf=MLPClassifier(alpha=1,hidden_layer_sizes=(100,100,100),max_iter=2000,random_state= 0)
get_model_performance(clf,X_train,Y_train,X_test,Y_test)
#clf.classes_


# In[10]:


def get_whole_model_performance(model,X,Y):
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model.fit(X_train,Y_train)
    print(cross_val_score(model, X, Y, cv=5))
    #print(model.score(X_test, Y_test))


# In[11]:


X_data=np.loadtxt(r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\data_X.txt')
Y_data=np.loadtxt(r'C:\Users\mahat\Documents\CSV_data_Tuesday_without_norm\CSV\data_Y.txt')
sm = SMOTE(random_state = 2) 
X_data, Y_data = sm.fit_sample(X_train, Y_train) 
clf=[DecisionTreeClassifier(criterion='entropy',max_depth=7,random_state=2),RandomForestClassifier(random_state=3,criterion='entropy',max_depth=11,max_features='auto',n_estimators=200), KNeighborsClassifier(n_neighbors=1,metric='manhattan',weights='uniform'),MLPClassifier(alpha=1,hidden_layer_sizes=(100,100,100),max_iter=2000,random_state= 0)]
i=0
for c in clf:
    get_whole_model_performance(c,X_data,Y_data)
    path=r'C:\Users\mahat\Documents\models'
    os.makedirs(path, exist_ok=True)
    file_name=path+'\model'+str(i)+'.pkl'
    #abspath = pathlib.Path(file_name).absolute()
    with open(file_name, 'wb') as f:
        pickle.dump(c, f)
    i+=1


# In[ ]:




