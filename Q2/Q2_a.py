#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
from tqdm import tqdm
from libsvm import svmutil
import time
import csv
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from libsvm import svmutil
from sklearn import svm
from sklearn.model_selection import cross_val_score
import sys

# In[ ]:


#paramters

p1 = sys.argv[1] #training file
p2 = sys.argv[2] # testing file
p3 = sys.argv[3]
# p1 = "fashion_mnist/train.csv" #training file
# p2 = "fashion_mnist/test.csv"  #testing file
# p3 = "fashion_mnist/val.csv"  #validation file


# In[ ]:


def read_data(p1,i,j):
    df_train_data = pd.read_csv(p1,encoding='latin-1', header=None)
    XY_train_78 = df_train_data[(df_train_data[784]==i) | (df_train_data[784]==j)]
    XY_train_78 = XY_train_78.reset_index(drop=True)
    XY_train_78.loc[XY_train_78[784] == i,784] = -1
    XY_train_78.loc[XY_train_78[784] == j,784] = 1
    XY_train = np.array(XY_train_78)
    X_train = XY_train[:,0:784]
    Y_train = XY_train[:,784:785]
    X_train_scaled = X_train/255
    
    return X_train_scaled,Y_train


# In[ ]:


m=0
n=0


# In[ ]:


def cal_k(x1):
    K = np.zeros((m,m))
    for i in tqdm(range(m)):
        for j in range(m):
            temp1 = np.linalg.norm(x1[i] - x1[j])
            temp2 = temp1**2
#             temp2 = temp1
            temp2 = 0.05*temp2
            temp2 = - temp2
            temp3 = np.exp(temp2)
            K[i][j] = temp3
    return K


# In[ ]:


def calc_params(x1,y1):
    C = 1
    y_mat = np.dot(y1,y1.T)
    K = cal_k(x1)
    print('shape of K',K.shape)
    
    H = y_mat*K*1.
    
    P = matrix(H)
    q = matrix(-np.ones((m, 1)))
    t1 = np.eye(m)
    t2 = (-1)*np.eye(m)
    G = matrix(np.vstack((t2,t1)))
    m_zeros = np.zeros(m)
    m_ones = np.ones(m)
    m_onesc = m_ones*C
    h_npstack = np.hstack((m_zeros, m_onesc))
    h = matrix(h_npstack)
    A = matrix(y1.reshape(1, -1))
    b = matrix(np.zeros(1))
    
    return P, q, G, h, A, b


# In[ ]:


def cal_alphas():
    P, q, G, h, A, b = calc_params(x1,y1)
    sol = cvxopt.solvers.qp(P, q, G, h, A, b) 
    return np.array(sol['x'])


# In[ ]:


def calc_nonzero_alphas(alphas_param):
    nonzero_alphas_list = []
    support_vectors = 0
    for i in alphas_param:
        if i > 1e-5 :
            nonzero_alphas_list.append(i)
            support_vectors +=1
        else :
            nonzero_alphas_list.append(0)
    print('No. of Support vectors :',support_vectors)
    return np.array(nonzero_alphas_list)


# In[ ]:


def cal_b(alpha_vlaues):
    for i in tqdm(range(m)):
        if alpha_vlaues[i]!=0:
            sum_val = 0
            for j in range(m):      
                t1 = alphas[j][0]*y1[j][0]
                t2 = np.linalg.norm(x1[i]-x1[j])
                t2 = t2**2
                t3 = -0.05*t2
                t4 = np.exp(t3)
                t5  =t1*t4
                sum_val = sum_val + t5
            alpha_vlaues[i] = y1[i][0] - sum_val

    return sum(alpha_vlaues)/np.count_nonzero(alpha_vlaues)


# In[ ]:


b_list = []
for i in range(9):
    for j in range(i+1,10):
        x1,y1 = read_data(p1,i,j)
        m,n= x1.shape
        P, q, G, h, A, b = calc_params(x1,y1)
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        alphas_reshape = alphas.reshape(m,)
        alpha_vlaues = calc_nonzero_alphas(alphas_reshape)
        finalb = cal_b(alpha_vlaues)
        print('value of b :',finalb)
        b_list.append(finalb)
print(b_list)


# In[ ]:


b_list[0]


# In[ ]:


def cal_accuracy(xtest,ytest):
    pred_list = []
    crct_cases=0
    n = len(xtest)
    alphasy1 = alphas.flatten()*y1.flatten()
    for i in tqdm(range(n)):
        neg=0
        pos = 0
        pred_val_list = []
        a1 = xtest[i]
        Kt = np.zeros(m)
        for i in range(m):
            t1 = np.linalg.norm(x1[i]-a1)
            t2 = t1**2
            Kt[i] = np.exp(-0.05*t2)
        
        pred_val_temp = np.matmul(alphasy1,Kt.T)
        
        for k in range(45):
            res = pred_val_temp + b_list[k]
            if res < 0:
                neg +=1
            elif res > 0:
                pos +=1
        
        if neg > pos:
            pred_list.append(-1)

        if pos  > neg:
            pred_list.append(1)
            
    
    for i in tqdm(range(n)):
        if ytest[i] == pred_list[i]:
            crct_cases +=1
    print('correct cases :',crct_cases)
    print('total test examples :',n)
    return crct_cases/n


# In[ ]:


for i in range(9):
    for j in range(i+1,10):
        xtest,ytest = read_data(p2,i,j)
        print('Accuracy on test data :',cal_accuracy(xtest,ytest)*100,'%')


# In[ ]:


xval,yval = read_data(p3)
print('Accuracy on validation data :',cal_accuracy(xval,yval)*100,'%')

