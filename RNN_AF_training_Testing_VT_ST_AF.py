#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import CSVLogger


# In[3]:


traindata = pd.read_csv('Traindata01.csv', header=None)
testdata = pd.read_csv('Testdata01.csv', header=None)


X = traindata.iloc[:,0:170]
Y = traindata.iloc[:,170]
C = testdata.iloc[:,170]
T = testdata.iloc[:,0:170]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


# In[4]:


batch_size = 4

# 1. define the network
model = Sequential()
model.add(SimpleRNN(64,input_dim=169))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Activation('sigmoid'))


# In[7]:
start_time = time.time()

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/rnn1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger=CSVLogger('logs/rnn1layer/rnntrainanalysis1.csv',separator=',',append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, callbacks=[checkpointer,csv_logger])
model.save("logs/rnn1layer/rnn1layer_model.hdf5")

end_time=time.time()
print("Training Time--- %s seconds ---" % (end_time - start_time))

# In[ ]:

model=load_model('RNNcheckpoint-298.hdf5')


start_time1 = time.time()
y_pred = model.predict(X_test,batch_size=1000)
np.savetxt('./OUTPUTS/RNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/RNNpredicted.txt', y_pred.round(), fmt='%01d')

#history=model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

end_time1=time.time()
print("Testing Time--- %s seconds ---" % (end_time1 - start_time1))



import pandas as pd
import time


# In[59]:


testdata2 = pd.read_csv("Ventriculartachycardia_dataset.csv",header=None)


# In[60]:


#X = traindata.iloc[:,0:169]
#Y = traindata.iloc[:,170]
C = testdata2.iloc[:,170]
T=testdata2.iloc[:,0:170]


#scaler = Normalizer().fit(X)
#trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

#y_train1 = np.array(Y)
y_test1 = np.array(C)

#y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

#X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))


# In[61]:


from keras.models import load_model


# In[62]:


start_time1 = time.time()

y_pred = model.predict(X_test,batch_size=1000)
np.savetxt('./OUTPUTS/RNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/RNNpredicted.txt', y_pred.round(), fmt='%01d')

#history=model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

end_time1=time.time()
print("Testing Time--- %s seconds ---" % (end_time1 - start_time1))


# In[63]:


y_pred2=np.argmax(y_pred,axis=1)

targets_test=np.argmax(y_test,axis=1)
    
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


# In[64]:


print(confusion_matrix(targets_test, y_pred2))

cnf_mat=confusion_matrix(targets_test, y_pred2)

for i in range(len(cnf_mat)):
    print("Accuracy of Class",i," = ","{0:.2f}".format(cnf_mat[i][i]/np.sum(cnf_mat[i])*100))


# In[85]:


tp=1068
tn=0
fp=0
fn=358

#specificity0 = tn/(tn+fp)
#print('class0',specificity0)
specificity1=tp/(tp+fn)
print('class1',specificity1)


# In[65]:


testdata3=pd.read_csv('ST_Data.csv')


# In[66]:


#X = traindata.iloc[:,0:170]
#Y = traindata.iloc[:,170]
C = testdata3.iloc[:,170]
T=testdata3.iloc[:,0:170]


#scaler = Normalizer().fit(X)
#trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

#y_train1 = np.array(Y)
y_test1 = np.array(C)

#y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

#X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))


# In[67]:


start_time1 = time.time()

y_pred = model.predict(X_test,batch_size=1000)
np.savetxt('./OUTPUTS/RNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/RNNpredicted.txt', y_pred.round(), fmt='%01d')

#history=model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

end_time1=time.time()
print("Testing Time--- %s seconds ---" % (end_time1 - start_time1))


# In[68]:


y_pred2=np.argmax(y_pred,axis=1)

targets_test=np.argmax(y_test,axis=1)
    
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


# In[69]:


print(confusion_matrix(targets_test, y_pred2))

cnf_mat=confusion_matrix(targets_test, y_pred2)

for i in range(len(cnf_mat)):
    print("Accuracy of Class",i," = ","{0:.2f}".format(cnf_mat[i][i]/np.sum(cnf_mat[i])*100))


# In[86]:


tp=31
tn=0
fp=0
fn=86

#specificity0 = tn/(tn+fp)
#print('class0',specificity0)
specificity1=tp/(tp+fn)
print('class1',specificity1)


# In[89]:


testdata4=pd.read_csv('VFecgch1_data.csv')


# In[90]:


#X = traindata.iloc[:,0:170]
#Y = traindata.iloc[:,170]
C = testdata4.iloc[:,170]
T=testdata4.iloc[:,0:170]


#scaler = Normalizer().fit(X)
#trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

#y_train1 = np.array(Y)
y_test1 = np.array(C)

#y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

#X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))


# In[91]:


start_time1 = time.time()

y_pred = model.predict(X_test,batch_size=1000)
np.savetxt('./OUTPUTS/RNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/RNNpredicted.txt', y_pred.round(), fmt='%01d')

#history=model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

end_time1=time.time()
print("Testing Time--- %s seconds ---" % (end_time1 - start_time1))


# In[92]:


y_pred2=np.argmax(y_pred,axis=1)

targets_test=np.argmax(y_test,axis=1)
    
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


# In[93]:


print(confusion_matrix(targets_test, y_pred2))

cnf_mat=confusion_matrix(targets_test, y_pred2)

for i in range(len(cnf_mat)):
    print("Accuracy of Class",i," = ","{0:.2f}".format(cnf_mat[i][i]/np.sum(cnf_mat[i])*100))


# In[94]:


tp=837
tn=0
fp=0
fn=107

#specificity0 = tn/(tn+fp)
#print('class0',specificity0)
specificity1=tp/(tp+fn)
print('class1',specificity1)


# In[95]:


testdata5=pd.read_csv('VFecgch2_data.csv')


# In[96]:


#X = traindata.iloc[:,0:170]
#Y = traindata.iloc[:,170]
C = testdata4.iloc[:,170]
T=testdata4.iloc[:,0:170]


#scaler = Normalizer().fit(X)
#trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

#y_train1 = np.array(Y)
y_test1 = np.array(C)

#y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

#X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))


# In[97]:


start_time1 = time.time()

y_pred = model.predict(X_test,batch_size=1000)
np.savetxt('./OUTPUTS/RNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/RNNpredicted.txt', y_pred.round(), fmt='%01d')

#history=model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

end_time1=time.time()
print("Testing Time--- %s seconds ---" % (end_time1 - start_time1))


# In[98]:


y_pred2=np.argmax(y_pred,axis=1)

targets_test=np.argmax(y_test,axis=1)
    
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


# In[99]:


print(confusion_matrix(targets_test, y_pred2))

cnf_mat=confusion_matrix(targets_test, y_pred2)

for i in range(len(cnf_mat)):
    print("Accuracy of Class",i," = ","{0:.2f}".format(cnf_mat[i][i]/np.sum(cnf_mat[i])*100))


# In[100]:


tp=837
tn=0
fp=0
fn=107

#specificity0 = tn/(tn+fp)
#print('class0',specificity0)
specificity1=tp/(tp+fn)
print('class1',specificity1)

