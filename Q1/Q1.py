#!/usr/bin/env python
# coding: utf-8

# #Part (A) Implementation of Naive Bayes Theorem
# #Que1(a),Que1(b),Que1(c)
# Run command : python Q1.py training_file_path testing_file_path

# In[57]:


#import all libraries here
import pandas as pd
import re 
from tqdm import tqdm
import math
from random import *
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import nltk
import sys
from nltk.tokenize import word_tokenize


# In[ ]:


p1 = sys.argv[1]
p2 = sys.argv[2]


# In[47]:


#paramters
# p0 = 1
# p1 = "trainingandtestdata/training.1600000.processed.noemoticon.csv" #training file
# p2 = "trainingandtestdata/testdata.manual.2009.06.14.csv"  #testing file


# In[32]:


#Reading data from the given data file into data frames
#0th (label) and 5th(tweets) are only extracted from the file
def read_data(file_path):
    df = pd.read_csv(file_path,encoding='latin-1', usecols=[0,5], header=None)
    return df


# In[33]:


df_data = read_data(p1)


# In[34]:


#Helping function to create a dictionary
#no_samples: how many records we need from the data file

def create_dict(df_input, no_samples):
    new_dict = {}
    for tweet in tqdm(df_input):
        tweet_list = tweet.split()
        for word in tweet_list:
            if(word in new_dict.keys()):
                new_dict[word]+=1
            else:
                new_dict[word]=1
    return new_dict
    


# In[35]:


#function to make two dictionaries one for each class 0 and 1
#it puts both dictionaries into a list and returns the list
#0 = negative, 4 = positive
def make_dictionaries():
    df_neg_data = df_data[df_data[0] == 0][5].tolist()
    df_pos_data = df_data[df_data[0] == 4][5].tolist()
    
    
    prob_pos_class = (len(df_pos_data)/(len(df_neg_data) + len(df_pos_data) ))
    prob_neg_class = (len(df_neg_data)/(len(df_neg_data) + len(df_pos_data) ))
        
    dict_neg_data = create_dict(df_neg_data,len(df_neg_data))
    dict_pos_data = create_dict(df_pos_data,len(df_pos_data))
    list_freq = [dict_neg_data,dict_pos_data]
    return list_freq,prob_pos_class,prob_neg_class
    


# In[36]:


#It takes a dictionary as input and returns another dictionary with probability of each word
# no of times a word occur / total no. of words (frequencies are considered not the unique words)
# Laplace_Smoothing is done here itself
def calculate_probs(input_dict,X):
    dict_prob={}
    alpha = len(X)
    c = 1
    total_words = sum(input_dict.values())
    print('alpha :',alpha,total_words)
    for i in X:
        val = input_dict.get(i,0)
        dict_prob[i]=((val + c)/(total_words + c*alpha))
        
    return dict_prob
    


# In[37]:


list_freq,prob_pos_class,prob_neg_class = make_dictionaries()


# In[38]:


#function to return the list of probability dictionaries
def return_prob_dict():
    X = (set(list_freq[0]) | set(list_freq[1]))
    print('no of words :',len(X))
    #build probabilities dictionary for negative class
    dict_neg_prob = calculate_probs(list_freq[0],X)
    #build probabilities dictionary for positive class
    dict_pos_prob = calculate_probs(list_freq[1],X)
    list_prob = [dict_neg_prob, dict_pos_prob]
    
    return list_prob
    
    


# In[39]:


#list of probabilities class wise
list_prob= return_prob_dict()


# In[40]:


X = (set(list_prob[0]) | set(list_prob[1]))
len(X)


# In[41]:


#returns the list of words of a sentence
def return_tokens(tweet):
    list_of_words = tweet.split()
    return list_of_words


# In[42]:


#calculates P(new_tweet/-ve)
def cal_prob_neg_class(new_tweet):
    prob = 0
    total_keys = len(list_prob[0])
    c = 1
    list_of_words = return_tokens(new_tweet)
    for word in list_of_words:
        prob += math.log(list_prob[0].get(word,1/(c*total_keys)))
    return prob


# In[43]:


#calculates P(new_tweet/+ve)
def cal_prob_pos_class(new_tweet):
    prob = 0
    total_keys = len(list_prob[1])
    list_of_words = return_tokens(new_tweet)
    c = 1
    for word in list_of_words:
        prob += math.log(list_prob[1].get(word,(c/(c*total_keys))))
    return prob


# #p(+ve/New_Tweet) = p(New_Tweet/+Ve)*p(+ve)
# #p(-ve/New_Tweet) = p(New_Tweet/-Ve)*p(+ve)

# In[44]:


#predict the class of new tweet
def make_prediction(new_tweet):
    prob_new_tweet_given_pos_class = cal_prob_pos_class(new_tweet)
    prob_new_tweet_given_neg_class = cal_prob_neg_class(new_tweet)
    
    pos_class_prob = prob_new_tweet_given_pos_class + math.log(prob_pos_class)
    neg_class_prob = prob_new_tweet_given_neg_class + math.log(prob_neg_class)
    
    if pos_class_prob > neg_class_prob:
        return 4
    else:
        return 0


# In[45]:


import warnings
warnings.filterwarnings("ignore")


# In[46]:


def calculate_accuracy(file_path,data_set):
    pred_list=[]
    if data_set == 'test':
        df_test_data = pd.read_csv(file_path,encoding='latin-1', usecols=[0,5], header=None)
    else :
        df_test_data = df_data
    test_data = df_test_data[df_test_data[0] !=2]
    test_data.reset_index(drop=True, inplace=True)
    X = test_data[5].tolist()
    for i in tqdm(X):
        pred=make_prediction(i)
        pred_list.append(pred)
    test_data['pred']=pred_list
    conf_matrix = confusion_matrix(test_data[0],test_data['pred'])
    accuracy = len(test_data[test_data[0]==test_data['pred']])/len(test_data)
    return accuracy,conf_matrix
    


# In[59]:


#Que1(a) Report accuracy over the training as well as the test set
print('--------------------Que1 (a) Accuracies over training and test data--------------------')


# In[48]:


accuracy,conf_matrix = calculate_accuracy(p2,'test')
print('Accuracy on Test Data :',accuracy)
print(conf_matrix)


# In[49]:


accuracy_train,conf_matrix_train = calculate_accuracy(p1,'train')
print('Accuracy on Training Data :',accuracy_train)
print(conf_matrix_train)


# In[ ]:


print('--------------------Que1 (c) Confusion Matrix--------------------')


# In[50]:


binary = conf_matrix
fig, ax = plot_confusion_matrix(conf_mat=binary,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()


# In[51]:


def calculate_rand_accuracy(file_path):
    rand_list=[]
    df_test_data = pd.read_csv(file_path,encoding='latin-1', usecols=[0,5], header=None)
    test_data = df_test_data[df_test_data[0] !=2]
    print("Number of examples for testing :",len(test_data))
    test_data.reset_index(drop=True, inplace=True)
    X = test_data[5]
    for j in range (len(test_data)):
        rand_list.append(sample([0,4],  1)[0] )
    
    test_data['rand']=rand_list
#     print(test_data)
    print('Number of value match :',len(test_data[test_data[0] == test_data['rand']]))
    return len(test_data[test_data[0] == test_data['rand']])/len(test_data)


# In[52]:


print('--------------------Que1 (b) Random/Majority Classifier--------------------')
print('Random classifier accuracy on test data :',calculate_rand_accuracy(p2))


# In[53]:


def calculate_major_accuracy(file_path,val):
    major_list=[]
    df_test_data = pd.read_csv(file_path,encoding='latin-1', usecols=[0,5], header=None)
    test_data = df_test_data[df_test_data[0] !=2]
    print("Number examples for testing :",len(test_data))
    test_data.reset_index(drop=True, inplace=True)
    X = test_data[5]
    for j in range (len(test_data)):
        major_list.append(val)
    
    test_data['major']=major_list
    print('Number of value match :',len(test_data[test_data[0] == test_data['major']]))
    return len(test_data[test_data[0] == test_data['major']])/len(test_data)


# In[54]:


print('Accuracy of majority classifier on test data as 0 as majority value :',calculate_major_accuracy(p2,0))


# In[55]:


print('Accuracy of majority classifier on test data as 4 as majority value :',calculate_major_accuracy(p2,4))


# In[ ]:




