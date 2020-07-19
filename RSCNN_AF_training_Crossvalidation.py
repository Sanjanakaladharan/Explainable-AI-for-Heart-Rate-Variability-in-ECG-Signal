#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.preprocessing import Normalizer
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D,Input,Activation,Add
from keras import callbacks
from sklearn.model_selection import KFold
#from keras.layers.advanced_activations import Softmax
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report,accuracy_score


# In[2]:


traindata = pd.read_csv('Traindata01.csv', header=None)
testdata = pd.read_csv('Testdata01.csv', header=None)


# In[3]:


traindata.head()


# In[4]:


print(np.shape(traindata))
X = traindata.iloc[:,0:170]
Y = traindata.iloc[:,171]
C = testdata.iloc[:,171]
T = testdata.iloc[:,0:170]
print(X.shape)
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))


# In[5]:


# Merge inputs and targets
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)


# In[6]:


inputs.shape


# In[7]:


seed = 7


# In[8]:


import time


# In[9]:


start_time = time.time()

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
acc_per_fold = []
loss_per_fold = []
f1_per_fold=[]
recall_per_fold=[]
precision_per_fold=[]
fold_no = 1
for train, test in kfold.split(inputs,targets):
  # create model
    inp=Input(shape=(170,1))
    c=Conv1D(filters=32,kernel_size=5,strides=1)(inp)

    c11=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(c)
    a11=Activation("relu")(c11)
    c12=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(a11)
    s11=Add()([c12,c])
    a12=Activation("relu")(s11)
    m11=MaxPooling1D(pool_size=5,strides=2)(a12)

    c21=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(m11)
    a21=Activation("relu")(c21)
    c22=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(a21)
    s21=Add()([c22,m11])
    a22=Activation("relu")(s21)
    m21=MaxPooling1D(pool_size=5,strides=2)(a22)

    c31=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(m21)
    a31=Activation("relu")(c31)
    c32=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(a31)
    s31=Add()([c32,m21])
    a32=Activation("relu")(s31)
    m31=MaxPooling1D(pool_size=5,strides=2)(a32)

    c41=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(m31)
    a41=Activation("relu")(c41)
    c42=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(a41)
    s41=Add()([c42,m31])
    a42=Activation("relu")(s41)
    m41=MaxPooling1D(pool_size=5,strides=2)(a42)

    c51=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(m41)
    a51=Activation("relu")(c51)
    c52=Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(a51)
    s51=Add()([c52,m41])
    a52=Activation("relu")(s51)
    m51=MaxPooling1D(pool_size=5,strides=2)(a52)

    f1=Flatten()(m51)

    d1=Dense(32)(f1)
    a6=Activation("relu")(d1)
    d2=Dense(32)(a6)
    a7=Dense(2, activation="softmax")(d2)


    model=Model(inputs=inp,outputs=a7)


    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
    checkpointer = callbacks.ModelCheckpoint(filepath="./fold_no_"+str(fold_no)+"/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
    csv_logger = CSVLogger('./fold_no_'+str(fold_no)+'/rcnntrainanalysis1.csv',separator=',', append=False)
    # Fit the model
    model.fit(inputs[train], targets[train], epochs=100, batch_size=10, verbose=1,callbacks=[checkpointer,csv_logger])
    
     # evaluate the model
    scores = model.evaluate(inputs[test], targets[test], verbose=1)
    print('Score for fold '+ str(fold_no)+':'+str(model.metrics_names[0])+' of '+str(scores[0])+' ; '+str(model.metrics_names[1])+' of '+ str(scores[1]*100)+'%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    y_pred = model.predict(inputs[test])
    y_pred2=np.argmax(y_pred,axis=1)
    targets_test=np.argmax(targets[test],axis=1)
    
    print(classification_report(targets_test, y_pred2))
    accuracy = accuracy_score(targets_test, y_pred2)
    recall = recall_score(targets_test, y_pred2)
    precision = precision_score(targets_test, y_pred2)
    f1 = f1_score(targets_test, y_pred2)
    
    
    print("accuracy")
    print("%.3f" %accuracy)
    print("precision")
    print("%.3f" %precision)
    print("recall")
    print("%.3f" %recall)
    print("f1score")
    print("%.3f" %f1)
    
    f1_per_fold.append(f1)
    recall_per_fold.append(recall)
    precision_per_fold.append(precision)

    fold_no = fold_no + 1
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold ')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print('> Fold '+ str(i+1)+'- Loss: '+str(loss_per_fold[i])+' - Accuracy: '+str(acc_per_fold[i])+' %')
    print('precision: '+str(precision_per_fold[i])+' -recall: '+str(recall_per_fold[i])+' -f1: '+str(f1_per_fold[i]))
print('------------------------------------------------------------------------')
print('Average scores for all folds: ')
print('> Accuracy: '+str(np.mean(acc_per_fold))+' +- '+str(np.std(acc_per_fold)))
print('> Loss: '+str(np.mean(loss_per_fold)))
print('>precision: '+str(np.mean(precision_per_fold)))
print('>recall: '+str(np.mean(recall_per_fold)))
print('>f1: '+str(np.mean(f1_per_fold)))
print('------------------------------------------------------------------------')


print("--- %s seconds ---" % (time.time() - start_time))


