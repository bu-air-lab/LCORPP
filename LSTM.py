"""# ######################Layout of LAyers: incompleted#########################################"""

from __future__ import division
#import customscale as cs
from keras.models import Sequential 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.models import Model
from keras.layers import Dropout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding,Dense,LSTM, SimpleRNN
from keras import initializers
import pandas as pd
from sklearn.model_selection import train_test_split                #useful for splitting data into training and test sets
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer

from keras.preprocessing import sequence
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import *



seed=7
np.random.seed(seed)

'''
dataset=pd.DataFrame(pd.read_csv('xonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
X =data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(X)
X[allnan]=0
X= cs.custscale(X,2,10)
'''
dataset=pd.DataFrame(pd.read_csv('batch1x.csv',header=None))
data=np.array(dataset)

X =data[:,1:-1]    

scaler = preprocessing.MinMaxScaler(feature_range=(-6,2))
X= scaler.fit_transform(X)

print ('Xshape is')
print (X.shape)
'''
dataset=pd.DataFrame(pd.read_csv('yonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
Y =data[1:,1:]           #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(Y)
Y[allnan]=0
Y= cs.custscale(Y,2,7)
'''
dataset=pd.DataFrame(pd.read_csv('batch1y.csv',header=None))
data=np.array(dataset)

Y =data[:,1:-1]    

scaler = preprocessing.MinMaxScaler(feature_range=(-1,2.5))
Y= scaler.fit_transform(Y)

print ('Yshape is')
print (Y.shape)

'''
dataset=pd.DataFrame(pd.read_csv('mthetaonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
mtheta =data[1:,1:]           #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(mtheta)
mtheta[allnan]=0
mtheta= cs.custscale(mtheta,2,14)
'''
#dataset=pd.DataFrame(pd.read_csv('intershortmtheta30.csv',header=None))
#data=np.array(dataset)

#mtheta =data[:,1:-1]    

#scaler = preprocessing.MinMaxScaler(feature_range=())
#mtheta= scaler.fit_transform(mtheta)

#print ('mthetashape is')
#print mtheta.shape

'''
vec=pd.DataFrame(pd.read_excel('veclength.xlsx',sheetname='Sheet1'))
vec=np.array(vec)
Labels=vec[:X.shape[0],-1:]
'''
Labels=data[:,-1:]
(m,n)=X.shape
num_features=2
#token=int(n/4)
token=n
newarray=np.empty([m,num_features*token])

for i in range(token):
	newarray[:,num_features*i]=X[:,i]
	newarray[:,num_features*i+1]=Y[:,i]
  #  newarray[:,num_features*i+2]=mtheta[:,i]
  
#newarray=np.concatenate((data[:,0:1],newarray),axis=1)
(p,q)=newarray.shape

print (newarray.shape)

#scaler=MinMaxScaler()

#newarray =scaler.fit_transform(newarray)


newarray_train, newarray_test, Labels_train, Labels_test=train_test_split(newarray,Labels, test_size=0.4, random_state=42 )

Results=pd.DataFrame()
i=len(Results.index)

ep=300
newarray_train = newarray_train.reshape(newarray_train.shape[0],token, num_features)
newarray_test = newarray_test.reshape(newarray_test.shape[0],token, num_features)
print('newarray_train reshaped is')
print (newarray_train.shape)
print ('Labels train shape')
print (Labels_train.shape)
bs=32
for hidden_size in [100]:
#for hidden_size in [30]:
	Results.set_value(i,'Training',newarray_train.shape[0])
	Results.set_value(i,'test',newarray_test.shape[0])
	Results.set_value(i,'Input size','dropout=0.2 x,y short30')
	model=Sequential()
	model.add(LSTM(hidden_size,input_shape=(token,num_features)))
	model.add(Dropout(0.2))
	Results.set_value(i,'inputs features',num_features)
	Results.set_value(i,'hidden size',hidden_size)
	model.add(Dense(1,activation='sigmoid'))

	#start=time.time()
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	history=model.fit(newarray_train,Labels_train,validation_split=0.2,epochs=ep,batch_size=bs,shuffle=False)

	Results.set_value(i,'epochs',ep)
	Results.set_value(i,'batch size',bs)
	scores = model.evaluate(newarray_test, Labels_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	
	predictions = model.predict(newarray_test, verbose=1)

	print ('predictions ')
	#print predictions
	print ('predictions shape is') 
	print (predictions.shape)
	print (' ')
	model.save('iter%i.h5'%i)
	Results.set_value(i,'Saved model name','batch1.h5 ')

	binarizer = Binarizer(threshold=0.2).fit(predictions)
	binary1 = binarizer.transform(predictions)
	p2=precision_score(Labels_test, binary1, average='binary')
	print ('\n\nprecision with threshold 0.2')
	print (p2)
	Results.set_value(i,'precision on Test set 0.2',p2)
	

	r2=recall_score(Labels_test, binary1, average='binary')
	print ('\nrecall with threshhold 0.2')
	print (r2)
	Results.set_value(i,'recall on test set 0.2',r2)

	binarizer = Binarizer(threshold=0.5).fit(predictions)
	binary2 = binarizer.transform(predictions)

	p5=precision_score(Labels_test, binary2, average='binary')
	print ('\n\nprecision with threshold 0.5')
	print (p5)
	Results.set_value(i,'precision on Test set 0.5',p5)
	r5=recall_score(Labels_test, binary2, average='binary')
	print ('\nrecall with threshhold 0.5')
	print (r5)
	Results.set_value(i,'recall on test set 0.5',r5)

	binarizer = Binarizer(threshold=0.7).fit(predictions)
	binary3 = binarizer.transform(predictions)

	p7=precision_score(Labels_test, binary3, average='binary')
	print ('\n\nprecision with threshold 0.7')
	print (p7)
	Results.set_value(i,'precision on Test set 0.7',p7)
	r7=recall_score(Labels_test, binary3, average='binary')
	print ('\nrecall with threshhold 0.7')
	print (r7)
	Results.set_value(i,'recall on test set 0.7',r7)

	model.summary()
	model.get_config()
	#writer = pd.ExcelWriter('batch1.xlsx')
	#Results.to_excel(writer,'Sheet1')
	#writer.save()


		# Plot loss histroy
		# summarize history for loss
	'''	
	fig=plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()
	fig.savefig('loss history of model %i'%i)
		
	fig=plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	fig.savefig('accuracy history of model %i'%i) 
	'''	 
	i=i+1
	model=None

