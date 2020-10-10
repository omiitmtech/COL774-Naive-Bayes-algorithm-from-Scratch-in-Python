#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
from tqdm import tqdm
from libsvm import svmutil
import time
import csv
import sys 


# In[3]:

#python Q1_SVM "fashion_mnist/train.csv" "fashion_mnist/test.csv" "fashion_mnist/val.csv"
#paramters
p1 = sys.argv[1] #training file
p2 = sys.argv[2]  #testing file
p3 = sys.argv[3]  #validation file

# p1 = "fashion_mnist/train.csv" #training file
# p2 = "fashion_mnist/test.csv"  #testing file
# p3 = "fashion_mnist/val.csv"  #validation file


# In[4]:


def read_data(p1):
    df_train_data = pd.read_csv(p1,encoding='latin-1', header=None)
    XY_train_78 = df_train_data[(df_train_data[784]==7) | (df_train_data[784]==8)]
    XY_train_78 = XY_train_78.reset_index(drop=True)
    XY_train_78.loc[XY_train_78[784] == 7,784] = -1
    XY_train_78.loc[XY_train_78[784] == 8,784] = 1
    XY_train = np.array(XY_train_78)
    X_train = XY_train[:,0:784]
    Y_train = XY_train[:,784:785]
    X_train_scaled = X_train/255
    
    return X_train_scaled,Y_train


# In[5]:


x1,y1 = read_data(p1)
m,n= x1.shape
print('shape of x1 :',x1.shape)
print('shape of y1 :',y1.shape)


# In[6]:


def calc_params(x1,y1):
    C = 1
    y1 = y1.reshape(-1,1) 
    xprime = y1*x1
    xprime_T = xprime.T
    H = np.dot(xprime , xprime_T)
    P = matrix(H)
    q = matrix(-np.ones((m, 1)))
    
    s1 = np.eye(m)
    s2 = s1*(-1)
    s3 = np.vstack(s2,s1)
    G = matrix(s3)

    m_zeros = np.zeros(m)
    m_ones = np.ones(m)
    m_onesc = m_ones*C
    h_npstack = np.hstack((m_zeros, m_onesc))
    h = matrix(h_npstack)
    A = matrix(y1.reshape(1, -1)) 
    b = matrix(np.zeros(1))
    
    return P, q, G, h, A, b


# In[7]:


def cal_alphas():
    P, q, G, h, A, b = calc_params(x1,y1)
    sol = cvxopt.solvers.qp(P, q, G, h, A, b) 
    return np.array(sol['x'])


# In[8]:


t1 = time.time()
alphas = cal_alphas()
print('\n\nTraining time in seconds = ',time.time()-t1)


# In[9]:


def calc_nonzero_alphas(alphas_param):
    nonzero_alphas_list = []
    support_vectors = 0
    for i in alphas:
        if i > 1e-5 :
            nonzero_alphas_list.append(i)
            support_vectors +=1
        else :
            nonzero_alphas_list.append(0)
    print('No. of Support vectors :',support_vectors)
    return nonzero_alphas_list


# In[10]:


alphas = alphas.reshape(m,)


# In[11]:


#updating alpha values with 0 where alpha_i is < 1e-5
#It prints the number of support vectors
alpha_vlaues = calc_nonzero_alphas(alphas)


# In[12]:


#function which clculates w
#w = sigmai=1 to m, alpha_i*y^(i)*x^(i)
def calc_w(p_alpha_vals,p_x_vals,p_y_vals):
    print('Shape of alpha values :',p_alpha_vals.shape)
    print('Shape of x values :',p_x_vals.shape)
    print('Shape of y values :',p_y_vals.shape)
    p_alpha_vals = p_alpha_vals.reshape(m,1)
    p_y_vals = p_y_vals.reshape(m,1)
    mul_res = p_alpha_vals * p_y_vals * p_x_vals #alpha_i*y^(i)*x^(i)
    w_val = np.sum(mul_res,axis=0) #rowwise sum
    return w_val


# In[13]:


def cal_b_val():
    min_val = 1000.0
    max_val = - 1000.0
    w = calc_w(alphas,x1,y1).reshape(n,1)
    w_trans = np.transpose(w) #w_trans = 1x784
    XY = np.hstack((x1,y1))
    print(XY.shape)
    for i in range(m):
        if y1[i] == -1:
            temp_val = x1[i].reshape(n,1)
            mul = np.dot(w_trans,temp_val)
            if mul > max_val:
                max_val = mul
        elif y1[i] == 1:
            temp_val = x1[i].reshape(n,1)
            mul = np.dot(w_trans,temp_val)
            if mul < min_val:
                min_val = mul
    print('min val :',min_val)
    print('max val :',max_val)
    
    return w_trans,-(max_val + min_val)/2


# In[14]:


w_trans,b = cal_b_val()


# In[26]:


print('The mean of W :',np.mean(w_trans))
print('Value of b :',b[0])


# In[16]:


def cal_accuracy(xtest,ytest):
    n = len(xtest)
    crct_cases = 0
    pred_list = []
    for i in tqdm(range(n)):
        temp1 = xtest[i].reshape(784,1)
        temp2 = np.dot(w_trans,temp1)
        pred_val = temp2 + b
        
        if pred_val < 0 :
                pred_list.append(-1)
        elif pred_val > 0:
                pred_list.append(1)
                
    for i in tqdm(range(n)):
        if ytest[i] == pred_list[i]:
            crct_cases +=1
    print('correct cases :',crct_cases)
    print('total test examples :',n)
    return crct_cases/n   


# In[17]:


xtest,ytest = read_data(p2)
print('Accuracy on test data :',cal_accuracy(xtest,ytest)*100,'%')


# In[18]:


xval,yval = read_data(p3)
print('Accuracy on validation data :',cal_accuracy(xval,yval)*100,'%')


# In[20]:


xtrain,ytrain = read_data(p1)
print('Accuracy on training data :',cal_accuracy(xtrain,ytrain)*100,'%')


# In[ ]:




