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
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from libsvm import svmutil
from sklearn import svm
from sklearn.model_selection import cross_val_score


# In[2]:


#paramters
p1 = sys.argv[1] #training file
p2 = sys.argv[2]  #testing file
p3 = sys.argv[3]  #validation file
# p1 = "fashion_mnist/train.csv" #training file
# p2 = "fashion_mnist/test.csv"  #testing file
# p3 = "fashion_mnist/val.csv"  #validation file


# In[3]:


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


# In[4]:


def read_data_nofilter(p1):
    df_train_data = np.array(pd.read_csv(p1,encoding='latin-1', header=None))
    X_train = df_train_data[:,0:784]
    Y_train = df_train_data[:,784:785]
    X_train_scaled = X_train/255
    
    return X_train_scaled,Y_train


# In[5]:


x1,y1 = read_data(p1)
m,n= x1.shape
print('shape of x1 :',x1.shape)
print('shape of y1 :',y1.shape)


# In[6]:


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


# In[7]:


def calc_params(x1,y1):
    C = 1
    y_mat = np.dot(y1,y1.T)
    K = cal_k(x1)
    print('shape of K',K.shape)
    
    H = y_mat*K*1.
    
    P = matrix(H)
    q = matrix(-np.ones((m, 1)))
    t1 = np.eye(m)
    t1 = (-1)*t1
    G = matrix(np.vstack((t1,np.eye(m))))
    m_zeros = np.zeros(m)
    m_ones = np.ones(m)
    m_onesc = m_ones*C
    h_npstack = np.hstack((m_zeros, m_onesc))
    h = matrix(h_npstack)
    A = matrix(y1.reshape(1, -1))
    b = matrix(np.zeros(1))
    
    return P, q, G, h, A, b


# In[8]:


def cal_alphas():
    P, q, G, h, A, b = calc_params(x1,y1)
    sol = cvxopt.solvers.qp(P, q, G, h, A, b) 
    return np.array(sol['x'])


# In[9]:


t0 = time.time()
alphas = cal_alphas()
print('Time to calculare alphas in seconds = ',time.time()-t0)


# In[10]:


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


# In[11]:


alphas_reshape = alphas.reshape(m,)


# In[12]:


#updating alpha values with 0 where alpha_i is < 1e-5
#It prints the number of support vectors
alpha_vlaues = calc_nonzero_alphas(alphas_reshape)


# In[13]:


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


t0 = time.time()
finalb = cal_b(alpha_vlaues)
print('\n\nTraining time in seconds = ',time.time()-t0)


# In[ ]:


def cal_accuracy(xtest,ytest):
    pred_list = []
    crct_cases=0
    n = len(xtest)
    alphasy1 = alphas.flatten()*y1.flatten()
    for i in tqdm(range(n)):
        a1 = xtest[i]
        Kt = np.zeros(m)
        for i in range(m):
            t1 = np.linalg.norm(x1[i]-a1)
            t2 = t1**2
            Kt[i] = np.exp(-0.05*t2)
        
        pred_val = np.matmul(alphasy1,Kt.T)+finalb
        if pred_val < 0:
            pred_list.append(-1)

        if pred_val > 0:
            pred_list.append(1)
    
    for i in tqdm(range(n)):
        if ytest[i] == pred_list[i]:
            crct_cases +=1
    print('correct cases :',crct_cases)
    print('total test examples :',n)
    return crct_cases/n


# In[ ]:


xtest,ytest = read_data(p2)
print('Accuracy on test data :',cal_accuracy(xtest,ytest)*100,'%')


# In[ ]:


xval,yval = read_data(p3)
print('Accuracy on validation data :',cal_accuracy(xval,yval)*100,'%')


# In[ ]:


xtrain,ytrain = read_data(p1)
print('Accuracy on test data :',cal_accuracy(xtrain,ytrain)*100,'%')


# #Part_B_Q1.C

# In[ ]:


xtrain,ytrain = read_data(p1)
ytrain = ytrain.reshape(m,)
linear_model = SVC(kernel='linear')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'linear_model.fit(xtrain, ytrain)')


# In[ ]:


xtest,ytest = read_data(p2)
xval,yval = read_data(p3)


# In[ ]:


print('Mean of W of the linear SVM Model :',np.mean(linear_model.coef_))
print('b value of the linear SVM Model :',linear_model.intercept_[0])
print('No. of support vectors of the linear SVM Model :',linear_model.n_support_[0] + linear_model.n_support_[1])
print('Accuracy of Test data on the linear SVM Model :',linear_model.score(xtest,ytest)*100,'%')
print('Accuracy of Validation data on the linear SVM Model :',linear_model.score(xval,yval)*100,'%')


# In[ ]:


Gaussian_model = SVC(kernel='rbf',gamma=0.05)


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'Gaussian_model.fit(xtrain, ytrain)')


# In[ ]:


# print('Mean of W of the gaussian SVM Model :',np.mean(Gaussian_model.coef_))
print('b value of the gaussian SVM Model :',Gaussian_model.intercept_[0])
print('No. of support vectors of the gaussian SVM Model :',Gaussian_model.n_support_[0] + linear_model.n_support_[1])
print('Accuracy of Test data on the gaussian SVM Model :',Gaussian_model.score(xtest,ytest)*100,'%')
print('Accuracy of Validation data on the gaussian SVM Model :',Gaussian_model.score(xval,yval)*100,'%')


# In[ ]:


#Que2.b
#Multi Class SVM
x_train_nf,y_train_nf = read_data_nofilter(p1)
xtest_nf,ytest_nf = read_data_nofilter(p2)
xval_nf,yval_nf = read_data_nofilter(p3)


# In[ ]:


print('Que_2(b)Learning with Gaussian Kernel')
multi_model = SVC(kernel='rbf',gamma=0.05)
t0 = time.time()
multi_model.fit(x_train_nf, y_train_nf)
print('Training time in seconds = ',time.time()-t0)

print('Accuracy on test data :',multi_model.score(xtest_nf, ytest_nf))
print('Accuracy on validation data :',multi_model.score(xval_nf, yval_nf))


# In[ ]:


svm_predictions_mul = multi_model.predict(xtest_nf) 


# In[ ]:


ConfusionMatrix2 = confusion_matrix(yval_nf,yval_predicted)
print('\n\nConfusion matrix for test data(part a\n\n')
print(ConfusionMatrix2)
ConfusionMatrix2 = confusion_matrix(yval_nf,yval_predicted)

ConfusionMatrix1 = confusion_matrix(ytest_nf,svm_predictions_mul)
print('\n\nConfusion matrix for test data\n\n')
print(ConfusionMatrix1)


# In[ ]:




