#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
df = pd.read_csv('bank-full.csv', sep = ';')
df.head()


# In[28]:


df.info()


# In[29]:


df.isna().sum()


# In[30]:


df.head()


# In[31]:


df['y']=df['y'].map({'yes':1,'no':0})
df2=pd.get_dummies(df,drop_first=True)
X=df2.drop(['y'], axis=1)
y=df2['y']


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[33]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[43]:


from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

model = SVC()
model.fit(X_train, y_train)
model.score(X_test ,y_test)
pred=model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))
plot_confusion_matrix(model,X_test,y_test)



plt.figtext(0.05,-0.05,"Observation: Support Vector Machine Classifier performed well with Accuracy score of 91%",
           family='Serif', size=14, ha='left', weight='bold')


# In[44]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix

model2 = GaussianNB()
model2.fit(X_train, y_train)
model2.score(X_test ,y_test)
pred2=model2.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test,pred2))
print(classification_report(y_test,pred2))
plot_confusion_matrix(model2,X_test,y_test)


plt.figtext(0.05,-0.05,"Observation: Naive Bayes Classifier performed well with Accuracy score of 86%",
           family='Serif', size=14, ha='left', weight='bold')


# In[ ]:
