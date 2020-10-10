#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile, f_classif
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import chi2
from tqdm import tqdm
import numpy as np
import sys

# In[2]:


p1 = sys.argv[1] #training file
p2 = sys.argv[2] # testing file
# p1 = "trainingandtestdata/training.1600000.processed.noemoticon.csv" #training file
# p2 = "trainingandtestdata/testdata.manual.2009.06.14.csv"  #testing file


# In[3]:


df_train_data = pd.read_csv(p1,encoding='latin-1', usecols=[0,5], header=None)
 


# In[4]:


df_test_data  = pd.read_csv(p2,encoding='latin-1', usecols=[0,5], header=None)


# In[5]:


features_train = df_train_data[5].tolist()
labels_train = df_train_data[0].tolist()

df_test_data = df_test_data[df_test_data[0]!=2]
features_test = df_test_data[5]
labels_test = df_test_data[0]


# In[6]:


vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english',dtype=np.float32)
X_train = vectorizer.fit_transform(features_train)
X_test = vectorizer.transform(features_test)


# In[7]:


def data_slice(X,Y,i,batch_size):
    return X[i:i+batch_size], Y[i:i+batch_size]


# In[8]:


model1 = GaussianNB()


# In[9]:


def train_model(train_data_parm, model_name):
    batch_size = 1000
    for i in tqdm(range(1600)):
        X_slice,Y_slice  = data_slice(train_data_parm,labels_train,i*batch_size,batch_size)
        model_name.partial_fit(X_slice.todense(),Y_slice,classes=[0,4])
    


# In[ ]:


t0 = time()
train_model(X_train,model1)
print(f"Time to train the model without features selection: {round(time()-t0, 3)}s")


# In[ ]:


def cal_accuracy(model_name,test_data):
    score_test = model_name.score(test_data.todense(), labels_test)
    print('shape of test data :',test_data.shape)
    print("Testing data score:", score_test)


# In[ ]:


t0 = time()
cal_accuracy(model1,X_test)
print(f"Prediction time (test): {round(time()-t0, 3)}s")


# In[ ]:


sp_model = SelectPercentile(chi2, percentile=10)
X_new_train = sp_model.fit_transform(X_train,labels_train)
X_new_test = sp_model.transform(X_test)


# In[ ]:


model2 = GaussianNB()


# In[ ]:


t0 = time()
train_model(X_new_train,model2)
print(f"Time to train model with features selection {round(time()-t0, 3)}s")


# In[ ]:


t0 = time()
cal_accuracy(model2,X_new_test)
print(f"Prediction time (test): {round(time()-t0, 3)}s")


# In[ ]:




