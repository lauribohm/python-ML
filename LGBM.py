#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
import lightgbm as lgb
from platform import python_version
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[3]:


df = pd.read_csv('bank-full.csv', sep = ';')
df.head()


# In[4]:


df.drop(['day','month'], axis=1, inplace =True)
df.head()


# In[5]:


y = df['y']
X = df.drop('y', axis=1)

del df

for c in X.columns:
    col_type = X[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c] = X[c].astype('category')
X.dtypes


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)


# In[22]:




clf = lgb.LGBMClassifier(random_state=314, 
                         silent=True,
                         max_depth = 4,
                         gamma= 1.0,
                         learning_rate=0.2)

clf.fit(X_train, y_train,
        eval_metric ='aucpr')


# In[21]:


plot_confusion_matrix(clf,
                      X_test,
                      y_test,
                      values_format ='d')


# In[ ]:
