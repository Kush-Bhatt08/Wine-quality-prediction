#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the Dataset

# In[4]:


wine = pd.read_csv("C:\\Users\\DELL\\Desktop\\winequality-red.csv")
wine.head()


# # Describing the data

# In[5]:


wine.describe()


# # Information from the data

# In[6]:


wine.info()


# # Counting the no. of instances for each class

# In[7]:


wine['quality'].value_counts()


# # Plotting out the data

# In[8]:


fig = plt.figure(figsize=(15,10))

plt.subplot(3,4,1)
sns.barplot(x='quality',y='fixed acidity',data=wine)

#Quality is high when volatile acidity is less.
plt.subplot(3,4,2)
sns.barplot(x='quality',y='volatile acidity',data=wine)


#Quality is high when citric acid is high.
plt.subplot(3,4,3)
sns.barplot(x='quality',y='citric acid',data=wine)

plt.subplot(3,4,4)
sns.barplot(x='quality',y='residual sugar',data=wine)


#Quality is high when chlorides are less.
plt.subplot(3,4,5)
sns.barplot(x='quality',y='chlorides',data=wine)

plt.subplot(3,4,6)
sns.barplot(x='quality',y='free sulfur dioxide',data=wine)

plt.subplot(3,4,7)
sns.barplot(x='quality',y='total sulfur dioxide',data=wine)

plt.subplot(3,4,8)
sns.barplot(x='quality',y='density',data=wine)

plt.subplot(3,4,9)
sns.barplot(x='quality',y='pH',data=wine)

#Quality is high when sulphates are more.
plt.subplot(3,4,10)
sns.barplot(x='quality',y='sulphates',data=wine)


#Quality is high when alcohol is more.
plt.subplot(3,4,11)
sns.barplot(x='quality',y='alcohol',data=wine)

plt.tight_layout()
plt.savefig('output.jpg',dpi=1000)


# In[9]:


# Making two categories bad and good
ranges = (2,6.5,8) 
groups = ['bad','good']
wine['quality'] = pd.cut(wine['quality'],bins=ranges,labels=groups)


# In[10]:


#Alotting 0 to bad and 1 to good
le = LabelEncoder()
wine['quality'] = le.fit_transform(wine['quality'])
wine.head()


# In[11]:


wine['quality'].value_counts()


# # Balancing the two classes

# In[12]:


good_quality = wine[wine['quality']==1]
bad_quality = wine[wine['quality']==0]

bad_quality = bad_quality.sample(frac=1)
bad_quality = bad_quality[:len(good_quality)]

new_df = pd.concat([good_quality,bad_quality])
# Here shuffling  the data and take a 100% fraction of the data.
new_df = new_df.sample(frac=1)
new_df


# In[13]:


new_df['quality'].value_counts()


# In[14]:


new_df.corr()['quality'].sort_values(ascending=False)


# In[15]:


# splitting the data in train and test
from sklearn.model_selection import train_test_split

X = new_df.drop('quality',axis=1) 
y = new_df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[16]:


param = {'n_estimators':[100,200,300,400,500,600,700,800,900,1000]}

grid_rf = GridSearchCV(RandomForestClassifier(),param,scoring='accuracy',cv=10,)
grid_rf.fit(X_train, y_train)

print('Best parameters --> ', grid_rf.best_params_)

pred = grid_rf.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print(accuracy_score(y_test,pred))


# # Here I have used GridSearchCV with Random Forest to find the best value of the ‘n_estimators’ parameter.
# #Finally,ended up with an accuracy of 79.3% which is very good for this much small dataset.

# In[ ]:




