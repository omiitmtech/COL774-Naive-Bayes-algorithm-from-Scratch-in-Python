from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile, f_classif
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import chi2
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import numpy as np
import time
import re
import sys


p1 = sys.argv[1] #training file
p2 = sys.argv[2] # testing file
# p1 = "trainingandtestdata/training.1600000.processed.noemoticon.csv" #training file
# p2 = "trainingandtestdata/testdata.manual.2009.06.14.csv"  #testing file

df_train_data = pd.read_csv(p1,encoding='latin-1', usecols=[0,5], header=None)
df_test_data  = pd.read_csv(p2,encoding='latin-1', usecols=[0,5], header=None)

def clean_data(tweet):
  stopwords = nltk.corpus.stopwords.words('english')
  lemma = WordNetLemmatizer()
  tokens_list = re.split('\W+',tweet) 
  lower_list = [word.lower() for word in tokens_list if word.isalpha()]   
  txt_clean = " ".join([lemma.lemmatize(x) for x in lower_list if x not in stopwords])
  return txt_clean

#cleaning the training data
t1 = time.time()
df_train_data['clean_data'] = df_train_data[5].apply(lambda x : clean_data(x))
print('\n\nTime to clean training data = ',time.time()-t1)

#cleaning the testing data
t1 = time.time()
df_test_data['clean_data'] = df_test_data[5].apply(lambda x : clean_data(x))
print('\n\nTime to clean testing data = ',time.time()-t1)

#Converting cleand data into list for faster processing
features_train = df_train_data['clean_data'].tolist()
labels_train = df_train_data[0].tolist()

df_test_data = df_test_data[df_test_data[0]!=2]
features_test = df_test_data['clean_data']
labels_test = df_test_data[0]

vectorizer = TfidfVectorizer(sublinear_tf=True, dtype=np.float32)
X_train = vectorizer.fit_transform(features_train)
X_test = vectorizer.transform(features_test)

def data_slice(X,Y,i,batch_size):
    return X[i:i+batch_size], Y[i:i+batch_size]

def train_model(train_data_parm, model_name):
    batch_size = 1000
    for i in tqdm(range(1600)):
        X_slice,Y_slice  = data_slice(train_data_parm,labels_train,i*batch_size,batch_size)
        model_name.partial_fit(X_slice.todense(),Y_slice,classes=[0,4])

def cal_accuracy(model_name,test_data):
    score_test = model_name.score(test_data.todense(), labels_test)
    print('shape of test data :',test_data.shape)
    print("Testing data score:", score_test)

model1 = GaussianNB()

t0 = time.time()
train_model(X_train,model1)
print('Time to train the model without features selection = ',time.time()-t0)

#training the model on the data without features selection
t0 = time.time()
cal_accuracy(model1,X_test)
print('Prediction time on test without features selection = ',time.time()-t0)

sp_model = SelectPercentile(chi2, percentile=10)
X_new_train = sp_model.fit_transform(X_train,labels_train)
X_new_test = sp_model.transform(X_test)

#creating new model
model2 = GaussianNB()

#training the model on the data without features selection
t0 = time.time()
train_model(X_new_train,model2)
print('\n\nTime to train the model with features selection = ',time.time()-t0)

#testing accuracy of the model on the data with features selection
t0 = time.time()
cal_accuracy(model2,X_new_test)
print('\n\nTime to check accuracy on testing data with features selection = ',time.time()-t0)

