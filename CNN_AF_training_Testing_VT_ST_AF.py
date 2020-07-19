#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[45]:


traindata = pd.read_csv('Traindata01.csv', header=None)
testdata = pd.read_csv('Testdata01.csv', header=None)


# In[46]:


np.shape(traindata)


# In[47]:


X = traindata.iloc[:,0:169]
Y = traindata.iloc[:,170]
C = testdata.iloc[:,170]
T = testdata.iloc[:,0:169]


# In[48]:


X.shape


# In[49]:


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


# In[50]:


cnn = Sequential()
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(169, 1)))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(2, activation="softmax"))


# In[51]:


cnn.summary()


# In[52]:


import time


# In[10]:


start_time = time.time()
# train
# define optimizer and objective, compile cnn

cnn.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="./CNN_checkpoints/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('./CNN_checkpoints/CNNtrainanalysis.csv',separator=',', append=False)
cnn.fit(X_train, y_train, nb_epoch=1000,callbacks=[csv_logger,checkpointer])
cnn.save("./CNN_checkpoints/CNN_model.hdf5")
end_time=time.time()
print("Training Time--- %s seconds ---" % (end_time - start_time))


# In[13]:


print("Training Time--- %s seconds ---" % (end_time - start_time))


# In[8]:


from keras.models import load_model
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report,accuracy_score


# In[53]:


start_time1 = time.time()
model=load_model('./Time_analysis/CNN_checkpoints/checkpoint-993.hdf5')
y_pred = model.predict(X_test,batch_size=1000)
np.savetxt('./OUTPUTS/CNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/CNNpredicted.txt', y_pred.round(), fmt='%01d')

#history=model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

end_time1=time.time()
print("Testing Time--- %s seconds ---" % (end_time1 - start_time1))


# In[54]:


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


# In[55]:


print(confusion_matrix(targets_test, y_pred2))


# In[56]:


cnf_mat=confusion_matrix(targets_test, y_pred2)


# In[57]:


for i in range(len(cnf_mat)):
    print("Accuracy of Class",i," = ","{0:.2f}".format(cnf_mat[i][i]/np.sum(cnf_mat[i])*100))


# In[83]:


tp=1998
tn=16523
fp=350
fn=559

specificity0 = tn/(tn+fp)
print('class0',specificity0)
specificity1=tp/(tp+fn)
print('class1',specificity1)


# In[ ]:


specificity=(1998/(+350))
print(specificity)


# In[58]:


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
np.savetxt('./OUTPUTS/CNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/CNNpredicted.txt', y_pred.round(), fmt='%01d')

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
np.savetxt('./OUTPUTS/CNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/CNNpredicted.txt', y_pred.round(), fmt='%01d')

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
np.savetxt('./OUTPUTS/CNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/CNNpredicted.txt', y_pred.round(), fmt='%01d')

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
np.savetxt('./OUTPUTS/CNNexpected.txt', y_test1, fmt='%01d')
np.savetxt('./OUTPUTS/CNNpredicted.txt', y_pred.round(), fmt='%01d')

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

