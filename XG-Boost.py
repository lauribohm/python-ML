#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from platform import python_version
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# In[6]:


df = pd.read_csv('bank-full.csv', sep = ';')
df.head()


# In[7]:


df.drop(['day','month'], axis=1, inplace =True)
df.head()


# In[8]:


df['previously_contacted'] = df['pdays'].apply(lambda x: 'yes' if x == -1 else 'no')
df.head()


# In[9]:


df['default'].replace({'no': "0", 'yes': "1"}, inplace=True)
df['default'] = df['default'].astype(int)
df['housing'].replace({'no': "0", 'yes': "1"}, inplace=True)
df['housing'] = df['housing'].astype(int)
df['loan'].replace({'no': "0", 'yes': "1"}, inplace=True)
df['loan'] = df['loan'].astype(int)
df['previously_contacted'].replace({'no': "0", 'yes': "1"}, inplace=True)
df['previously_contacted'] = df['previously_contacted'].astype(int)
df['y'].replace({'no': "0", 'yes': "1"}, inplace=True)
df['y'] = df['y'].astype(int)

#df.head()


# In[10]:


X = df.drop('y', axis=1).copy()
y = df['y'].copy()
#X.head()


# In[11]:


X.dtypes


# In[12]:


X_encoded = pd.get_dummies(X,columns= ['job',
                          'marital',
                          'education',
                          'contact',
                          'poutcome',
                          ])
X_encoded.head()


# In[13]:


y.unique()


# In[14]:


Sum = sum(y)
Length = len(y)
print(Sum/Length)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, stratify = y)


# In[16]:



clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')
clf_xgb.fit(X_train,
            y_train,
            verbose = True,
            #use_label_encoder=False,
            early_stopping_rounds=20,
            eval_metric ='aucpr',
            eval_set = [(X_test,y_test)])


# In[18]:


plot_confusion_matrix(clf_xgb,
                      X_test,
                      y_test,
                      values_format ='d')


# In[19]:


param_grid = {
    'max-depth': [4,6,8],
    'learning_rate': [0.4,0.3,0.2],
    'min_child_weight': [11],
    'gamma': [0,0.25,1.0],
    'req_lamda': [0, 1.0, 10],
    'scale_pos_weight': [1,3,5],
    
}

optimal_params = GridSearchCV(
    estimator = xgb.XGBClassifier(objective = 'binary:logistic',
                                  subsample= 0.8,
                                  seed= 1337,
                                  colsample_bytree = 0.7
                                 ),
    param_grid = param_grid,
    scoring= 'roc_auc',
    verbose=0,
    n_jobs = 10,
    cv = 3

)

optimal_params.fit(X_train,
            y_train,
            verbose = False,
            early_stopping_rounds=10,
            eval_metric ='auc',
            eval_set = [(X_test,y_test)])
print(optimal_params.best_params_)


# In[40]:


clf_xgb2 = xgb.XGBClassifier(objective = 'binary:logistic',
                             gamma= 1.0,  
                             max_depth= 5,
                             learning_rate = 0.2,
                             min_child_weight= 10
                             #req_lamda= 0, 
                                )
clf_xgb2.fit(X_train,
            y_train,
            verbose = True,
            early_stopping_rounds=20,
            eval_metric ='aucpr',
            eval_set = [(X_test,y_test)])


# In[41]:


plot_confusion_matrix(clf_xgb2,
                      X_test,
                      y_test,
                      values_format ='d')


# In[ ]:
