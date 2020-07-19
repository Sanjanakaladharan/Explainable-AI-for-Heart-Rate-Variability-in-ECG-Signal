#!/usr/bin/env python
# coding: utf-8

# In[32]:


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
#from keras.datasets import imdb
from keras import backend as K
#from sklearn.cross_validation import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report,accuracy_score


# In[33]:


traindata = pd.read_csv('./Traindata01.csv', header=None)
testdata = pd.read_csv('./Testdata01.csv', header=None)


# In[34]:


np.shape(traindata)


# In[35]:


X = traindata.iloc[:,0:170]
Y = traindata.iloc[:,171]
C = testdata.iloc[:,171]
T = testdata.iloc[:,0:170]


# In[36]:


X.shape


# In[37]:


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


# In[38]:


# Merge inputs and targets
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)


# In[39]:


seed = 7


# In[40]:


import time


# In[41]:


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
    cnn = Sequential()
    cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(170, 1)))
    cnn.add(MaxPooling1D(pool_length=(2)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(2, activation="softmax"))
    # Compile model
    cnn.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
    
    checkpointer = callbacks.ModelCheckpoint(filepath="./fold_no_"+str(fold_no)+"/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
    csv_logger = CSVLogger('./fold_no_'+str(fold_no)+'/cnntrainanalysis1.csv',separator=',', append=False)
    # Fit the model
    cnn.fit(inputs[train], targets[train], epochs=970, batch_size=10, verbose=1,callbacks=[checkpointer,csv_logger])
    
    # evaluate the model
    scores = cnn.evaluate(inputs[test], targets[test], verbose=1)
    print('Score for fold'+ str(fold_no)+':'+str(cnn.metrics_names[0])+' of '+str(scores[0])+' ; '+str(cnn.metrics_names[1])+' of '+ str(scores[1]*100)+'%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    y_pred = cnn.predict(inputs[test])
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
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print('> Fold'+ str(i+1)+'- Loss: '+str(loss_per_fold[i])+'- Accuracy: '+str(acc_per_fold[i])+'%')
    print('precision:'+str(precision_per_fold[i])+'-recall:'+str(recall_per_fold[i])+'-f1:'+str(f1_per_fold[i]))
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print('> Accuracy:'+str(np.mean(acc_per_fold))+'+-'+str(np.std(acc_per_fold)))
print('> Loss: '+str(np.mean(loss_per_fold)))
print('>precision:'+str(np.mean(precision_per_fold)))
print('>recall:'+str(np.mean(recall_per_fold)))
print('>f1:'+str(np.mean(f1_per_fold)))
print('------------------------------------------------------------------------')


print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[ ]:





# In[ ]:




