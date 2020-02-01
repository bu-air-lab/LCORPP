import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'errorbar.capsize':2,
'legend.fontsize': 'x-large'})
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
import pandas as pd
import csv

def plotbatchesbar():
	df = pd.read_csv('batches.csv').drop(['Unnamed: 0'],axis=1)
	print (df)
	print (df['F1 Score']) 

	# data to plot
	n_groups = 3
	#means_frank = (90, 55, 40)
	#means_guido = (85, 62, 54)

	# create plot
	fig, ax = plt.subplots(figsize=(8,6))
	index = np.arange(n_groups)
	bar_width = 0.25
	opacity = 0.8

	rects1 = ax.bar(index,list(df['F1 Score']) , bar_width,
	alpha=opacity,
	color='b',
	label='F1 Score')

	rects2 = ax.bar(index + bar_width,list(df['Accuracy']) , bar_width,
	alpha=opacity,
	color='g',
	label='Accuracy')

	costslist = list(df['Cost'])

	for x in costslist:
		x=x/100.0

	ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis


	rects3 = ax2.bar(index + 2*bar_width,list(df['Cost']) , bar_width,
	alpha=opacity,
	color='r',
	label='Cost')

	ax.set_xlabel('Batches', fontsize=16)
	ax.set_ylim([0.6, 1])
	#plt.ylabel('Probabilites/Costs')
	ax2.set_ylabel('Cost', fontsize=16)
	ax.set_ylabel('Probabilites', fontsize=16)
	plt.title('Comparison of different batches')
	plt.xticks(index + bar_width, ('No relabling', 'half relabling ', 'Total relabling'),fontsize=16)
	#plt.legend()
	ax2.legend(loc =1)
	ax.legend()
	plt.tight_layout()
	plt.savefig('batches.pdf')
	#plt.show()

def plotbatchesscatter():
	df = pd.read_csv('batchescomparison.csv').drop(['Unnamed: 0'],axis=1)
	print (df)
	 

	ax=plt.subplot(1,1,1)
	xaxis=['1','2','3','4']
	##############  F1 Score ########################
	
	l4 = plt.plot(xaxis,df.iloc[0:4,0],marker='^',markersize=8,linestyle='-',label='LCORPP (Ours)',color='b')
	
	plt.errorbar(xaxis,df.iloc[0:4,0],yerr=df.iloc[0:4,1], linestyle="None",label=None,color='b')

	plt.xlabel('Batch', fontsize= 16)
	plt.ylabel('F1 Score', fontsize= 16)
	#ax=plt.subplot(1,2,2)

	'''
	##############   Cost  #################
	
	l4 = plt.plot(xaxis,df.iloc[0:3,7],marker='^',linestyle='-.',label='LCORPP (Ours)')
	
	plt.errorbar(xaxis,df.iloc[0:3,7],yerr=df.iloc[0:3,5], linestyle="None",label=None)
	ax.legend(loc='upper left', bbox_to_anchor=(-1.10, 1.15),  shadow=True, ncol=1)
	plt.ylabel('Cost (s)', fontsize= 16)
	plt.xlabel('Batch', fontsize= 16)
	'''
	plt.savefig('batches.pdf')
	#plt.show()
	


def plotcomparison():

	df = pd.read_csv('comparison.csv')#.drop(['Unnamed: 0'],axis=1)
	print (df)
	print (df['F1 Score']) 

	# data to plot
	n_groups = 6
	#means_frank = (90, 55, 40)
	#means_guido = (85, 62, 54)

	# create plot
	fig, ax = plt.subplots(figsize=(8,6))
	index = np.arange(n_groups)
	bar_width = 0.2
	opacity = 0.8

	rects1 = ax.bar(index,list(df['F1 Score']) , bar_width,
	alpha=opacity,
	color='b',
	label='F1 Score')

	#rects2 = ax.bar(index + bar_width,list(df['Accuracy']) , bar_width,
	#alpha=opacity,
	#color='g',
	#label='Accuracy')

	costslist = list(df['Cost'])

	for x in costslist:
		x=x/100.0

	ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis


	rects3 = ax2.bar(index + 2*bar_width,list(df['Cost'])  , bar_width,
	alpha=opacity,
	color='r',
	label='Cost')

	ax.set_xlabel('Strategy', fontsize=16)
	ax.set_ylim([0.5, 1])
	#plt.ylabel('Probabilites/Costs')
	ax2.set_ylabel('Cost', fontsize=16)
	ax.set_ylabel('F1 Score', fontsize=16)
	plt.title('Comparison of different batches')
	plt.xticks(index + bar_width, ('L', 'R', 'L+\nR','Planning','R+P','LCORPP (Ours)'),fontsize=16)
	#plt.xticks(index + bar_width, list(df['Unnamed: 0']) ,fontsize=16)

	#plt.legend()
	ax2.legend(loc =1)
	ax.legend(loc=2)
	plt.tight_layout()
	plt.savefig('compariosn.pdf')
	#plt.show()	


def plotcomparisontwosubplots():

	#fig=plt.figure()

	fig=plt.figure(figsize=(10,4.8))

	df = pd.read_csv('comparison.csv')#.drop(['Unnamed: 0'],axis=1)
	print (df)
	print (df['F1 Score']) 

	ax=plt.subplot(1,2,1)
	# data to plot
	n_groups = 6
	#means_frank = (90, 55, 40)
	#means_guido = (85, 62, 54)
	
	# create plot
	#fig, ax = plt.subplots(figsize=(8,6))
	index = np.arange(n_groups)
	bar_width = 0.15
	opacity = 0.7
	new_x = [i for i in index]
	rects1 = ax.bar(new_x,list(df['F1 Score']) , yerr= df.iloc[0:6,5],width=0.9,
	alpha=opacity,
	color='k',
	label='F1 Score',zorder=3)
	ax.yaxis.grid(which ='major',zorder=0,color='grey')
	#rects2 = ax.bar(index + bar_width,list(df['Accuracy']) , bar_width,
	#alpha=opacity,
	#color='g',
	#label='Accuracy')
	plt.xticks(new_x , ('L', 'R', 'L+R','P','R+P','(Ours)\nLCORPP'),fontsize=16)
	#plt.axhline()
	costslist = list(df['Cost'])

	for x in costslist:
		x=x/100.0

	ax2=plt.subplot(1,2,2)
	# data to plot
	n_groups = 3

	index = np.arange(n_groups)
	 
	new_x = [i for i in index]
	rects3 = ax2.bar(new_x,df.iloc[3:6,4], yerr= df.iloc[3:6,3],width =0.5,
	alpha=opacity,
	color='k',
	label='Cost',zorder=3)
	ax2.yaxis.grid(which ='major',zorder=0,color='grey')

	ax.set_xlabel('Strategy', fontsize=18)
	ax2.set_xlabel('Strategy', fontsize=18)

	ax.set_ylim([0.5, 1])
	ax2.set_ylim([6, 13])
	ax2.set_xlim([-0.5, 2.5])

	#plt.ylabel('Probabilites/Costs')
	ax2.set_ylabel('Cost', fontsize=18)
	ax.set_ylabel('F1 Score', fontsize=18)
	#plt.title('Pairwise comparison of various SDM paradigms')
	plt.xticks(new_x, ('P','R+P','(Ours)\nLCORPP'),fontsize=16)
	#plt.xticks(index + bar_width, list(df['Unnamed: 0']) ,fontsize=16)

	#plt.legend()
	#ax2.legend(loc =1)
	#ax.legend(loc=2)
	#plt.suptitle('Performance of various SDM paradigms')
	
	plt.tight_layout()
	#plt.savefig('compariosn.pdf')
	plt.show()	
	



def inaccuratekb():


	df = pd.read_csv('inaccuratekb.csv').drop(['Unnamed: 0'],axis=1)
	print (df)
	fig=plt.figure(figsize=(10,4.8))

	print (df.iloc[1,0:3])
	#l1 = plt.plot(['a','b','c'],df.loc[1,0:2],marker='*',linestyle='-',label='MOMDP(ours)')
	
	#for count,metric in enumerate(list(df)):
	#	pass
		#print metric
	
	ax=plt.subplot(1,2,1)
	xaxis=['Low','Medium','High']
	##############  F1 Score ########################
	l1 = plt.plot(xaxis,df.iloc[0,0:3],marker='*',markersize=8,linestyle='-.',label='R',color='m')
	l2 = plt.plot(xaxis,df.iloc[1,0:3],marker='D',markersize=8,linestyle=':',label='L+R',color='c')
	l3 = plt.plot(xaxis,df.iloc[2,0:3],marker='o',markersize=8,linestyle='--',label='R+P',color='r')
	l4 = plt.plot(xaxis,df.iloc[3,0:3],marker='^',markersize=8,linestyle='-',label='LCORPP (Ours)',color='b')
	print (df.iloc[0,13:16])
	plt.errorbar(xaxis,df.iloc[0,0:3],yerr=df.iloc[0,13:16], linestyle="None",label=None,color='m')
	plt.errorbar(xaxis,df.iloc[1,0:3],yerr=df.iloc[1,13:16], linestyle="None",label=None,color='c')
	plt.errorbar(xaxis,df.iloc[2,0:3],yerr=df.iloc[2,13:16], linestyle="None",label=None,color='r')
	plt.errorbar(xaxis,df.iloc[3,0:3],yerr=df.iloc[3,13:16], linestyle="None",label=None,color='b')

	plt.xlabel('Knowledge level', fontsize= 18)
	plt.ylabel('F1 Score', fontsize= 18)
	ax=plt.subplot(1,2,2)

	##############   Cost  #################
	l1 = plt.plot(xaxis,df.iloc[0,3:6],marker='*',markersize=8,linestyle='-.',label='R',color='m')
	l2 = plt.plot(xaxis,df.iloc[1,3:6],marker='D',markersize=8,linestyle=':',label='L+R',color='c')
	l3 = plt.plot(xaxis,df.iloc[2,3:6],marker='o',markersize=8,linestyle='--',label='R+P',color='r')
	l4 = plt.plot(xaxis,df.iloc[3,3:6],marker='^',markersize=8,linestyle='-',label='LCORPP (Ours)',color='b')
	plt.errorbar(xaxis,df.iloc[2,3:6],yerr=df.iloc[2,6:9], linestyle="None",label=None,color='r')
	plt.errorbar(xaxis,df.iloc[3,3:6],yerr=df.iloc[3,6:9], linestyle="None",label=None,color='b')
	ax.legend(loc='upper left', bbox_to_anchor=(-1.05, 1.15),  shadow=True, ncol=4)
	plt.ylabel('Cost', fontsize= 18)
	plt.xlabel('Knowledge level', fontsize= 18)
	#fig.tight_layout()
	#plt.savefig('inaccuratekb.pdf')
	plt.show()	
   
	''' 
		plt.ylabel(metric, fontsize= xfont)
		plt.xlim(xaxis[0]-0.5,xaxis[-1]+0.5)
		xleft , xright =ax.get_xlim()
		ybottom , ytop = ax.get_ylim()
		ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)


		#plt.xlabel('Number of Properties')
		
		




	
		
	fig.tight_layout()
	plt.show()
		#fig.savefig('Results_'+str(num_trials)+'_trials_'+str(num_props)+'_queries_ask_cost'+str(ask_cost)+'_max_cost_prob65.png')
	fig.savefig(filename.split('.')[0]+'.'+file_extension)

	'''

def partialsensor():


	df = pd.read_csv('partialsensor.csv').drop(['Unnamed: 0'],axis=1)
	print (df)
	fig=plt.figure(figsize=(10,4.5))

	print (df.iloc[1,0:3])
	#l1 = plt.plot(['a','b','c'],df.loc[1,0:2],marker='*',linestyle='-',label='MOMDP(ours)')
	
	#for count,metric in enumerate(list(df)):
	#	pass
		#print metric
	
	ax=plt.subplot(1,2,1)
	xaxis=['Partial','Full']
	##############  F1 Score ########################
	l1 = plt.plot(xaxis,df.iloc[0,0:2],marker='*',markersize=8,linestyle='-.',label='L',color='g')
	l2 = plt.plot(xaxis,df.iloc[1,0:2],marker='D',markersize=8,linestyle=':',label='L+R',color='c')

	l4 = plt.plot(xaxis,df.iloc[2,0:2],marker='^',markersize=8,linestyle='-',label='LCORPP (Ours)',color='b')

	plt.errorbar(xaxis,df.iloc[0,0:2],yerr=df.iloc[0,9:11], linestyle="None",label=None,color='g')
	plt.errorbar(xaxis,df.iloc[1,0:2],yerr=df.iloc[1,9:11], linestyle="None",label=None,color='c')
	plt.errorbar(xaxis,df.iloc[2,0:2],yerr=df.iloc[2,9:11], linestyle="None",label=None,color='b')
	ax.set_ylim([0.4, 1])

	plt.xlabel('Trajectory', fontsize= 16)
	plt.ylabel('F1 Score', fontsize= 16)

	ax.legend(loc='upper left', bbox_to_anchor=(0.50, 1.15),  shadow=True, ncol=3)
	ax=plt.subplot(1,2,2)

	##############   Cost  #################
	#l1 = plt.plot(xaxis,df.iloc[0,2:4],marker='*',linestyle='-',label='L')
	#l2 = plt.plot(xaxis,df.iloc[1,2:4],marker='D',linestyle=':',label='L+R')
	
	l4 = plt.plot(xaxis,df.iloc[2,2:4],marker='^',markersize=8,linestyle='-',label='LCORPP (Ours)',color='b')
	plt.errorbar(xaxis,df.iloc[2,2:4],yerr=df.iloc[2,4:6], linestyle="None",label=None,color='b')

	#ax.legend(loc='upper left', bbox_to_anchor=(-1.10, 1.15),  shadow=True, ncol=2)
	plt.ylabel('Cost', fontsize= 16)
	plt.xlabel('Trajectory', fontsize= 16)

	#plt.show()
	plt.savefig('partialsensor.pdf')	

def partialsensorsingleplot():


	df = pd.read_csv('partialsensor.csv').drop(['Unnamed: 0'],axis=1)
	print (df)
	fig=plt.figure()

	print (df.iloc[1,0:3])
	#l1 = plt.plot(['a','b','c'],df.loc[1,0:2],marker='*',linestyle='-',label='MOMDP(ours)')
	
	#for count,metric in enumerate(list(df)):
	#	pass
		#print metric
	
	ax=plt.subplot(1,1,1)
	xaxis=[0.5,2.5]
	##############  F1 Score ########################
	l1 = plt.plot(xaxis,df.iloc[0,0:2],marker='*',markersize=8,linestyle='-.',label='L',color='g')
	l2 = plt.plot(xaxis,df.iloc[1,0:2],marker='D',markersize=8,linestyle=':',label='L+R',color='c')

	l4 = plt.plot(xaxis,df.iloc[2,0:2],marker='^',markersize=8,linestyle='-',label='LCORPP (Ours)',color='b')
	ax.yaxis.grid(which ='major',zorder=0,color='grey')

	plt.errorbar(xaxis,df.iloc[0,0:2],yerr=df.iloc[0,9:11], linestyle="None",label=None,color='g')
	plt.errorbar(xaxis,df.iloc[1,0:2],yerr=df.iloc[1,9:11], linestyle="None",label=None,color='c')
	plt.errorbar(xaxis,df.iloc[2,0:2],yerr=df.iloc[2,9:11], linestyle="None",label=None,color='b')
	ax.set_ylim([0.4, 1])
	ax.set_xlim([0, 3])

	plt.xticks(xaxis , ('Partial','Full'),fontsize=16)
	plt.xlabel('Trajectory', fontsize= 16)
	plt.ylabel('F1 Score', fontsize= 16)

	ax.legend(loc='upper left', bbox_to_anchor=(-0.00, 1.15),  shadow=True, ncol=3)
	

	plt.show()
	#plt.savefig('partialsensor.pdf')


def main():

	#plotbatchesscatter()
	#inaccuratekb()
	partialsensorsingleplot()
	#plotcomparisontwosubplots()


if __name__ == '__main__':
	main()
