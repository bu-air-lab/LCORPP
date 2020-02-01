from keras.preprocessing import sequence
import pandas as pd
import numpy as np

from numpy import *

'''
dataset=pd.DataFrame(pd.read_csv('xonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
X =data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(X)
X[allnan]=0

dataset=pd.DataFrame(pd.read_csv('yonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
Y =data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(Y)
Y[allnan]=0



dataset=pd.DataFrame(pd.read_csv('mthetaonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
mtheta=data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(mtheta)
mtheta[allnan]=0
#mtheta= preprocessing.scale(mtheta)

dataset=pd.DataFrame(pd.read_csv('vonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
v=data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(v)
v[allnan]=0
#mtheta= preprocessing.scale(mtheta)


dataset=pd.DataFrame(pd.read_csv('othetaonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
otheta=data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(otheta)
otheta[allnan]=0
#mtheta= preprocessing.scale(mtheta)
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
'''
# This function works well if only we have negative and positive values in the array
def custscale(array,down,up):

#	print np.amin(array)
#	print np.amax(array)
	output=((up-down)/(np.amax(array)-np.amin(array)))*(array-np.amin(array))+down		
        for x in range(array.shape[0]):
			for y in range(array.shape[1]):
				if array[x,y]==0:
					output[x,y]=0
			
	return output
