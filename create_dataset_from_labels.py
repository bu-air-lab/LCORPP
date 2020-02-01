import json
import numpy as np
path = './'

filenamex = 'interposx.csv'
filenamey = 'interposy.csv'

xdata = np.genfromtxt(path+filenamex, delimiter=',')
ydata = np.genfromtxt(path+filenamey, delimiter=',')

(m,n)= xdata.shape
randrow =3
print (xdata[randrow,n-1])
with open('newdataset.json') as json_file:
    newlabels = json.load(json_file)
print (n)
a = np.zeros((1,n))
b = np.zeros((1,n))
for ID in newlabels.keys():
	 row = np.where(xdata[:,0]==float(ID))
	 newxcopy = xdata[row[0][0],0:n].reshape(1,-1)
	 newycopy = ydata[row[0][0],0:n].reshape(1,-1)
	 newxcopy[0,n-1] = newlabels[ID]
	 newycopy[0,n-1] = newlabels[ID]
	 a= np.concatenate((a,newxcopy))
	 b= np.concatenate((b,newycopy))

print (a)
print (a.shape)
np.savetxt("batch1x.csv", a, delimiter=",")
np.savetxt("batch1y.csv", a, delimiter=",")

print (type(xdata))
