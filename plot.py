import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'errorbar.capsize':2})
import pandas as pd
import csv

def plotbatches():
	df = pd.read_csv('results.csv').drop(['Unnamed: 0'],axis=1)
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


	rects3 = ax2.bar(index + 2*bar_width,list(df['Cost']) , bar_width,
	alpha=opacity,
	color='r',
	label='Cost')

	ax.set_xlabel('Strategy', fontsize=16)
	ax.set_ylim([0.5, 1])
	#plt.ylabel('Probabilites/Costs')
	ax2.set_ylabel('Cost', fontsize=16)
	ax.set_ylabel('F1 Score', fontsize=16)
	plt.title('Comparison of different batches')
	plt.xticks(index + bar_width, ('Learning', 'Reasoning', 'Learning+\nReasoning','Planning','CORPP','LeRDeM (ours)'),fontsize=16)
	#plt.xticks(index + bar_width, list(df['Unnamed: 0']) ,fontsize=16)

	#plt.legend()
	ax2.legend(loc =1)
	ax.legend(loc=2)
	plt.tight_layout()
	plt.savefig('compariosn.pdf')
	#plt.show()	




def inaccuratekb():


	df = pd.read_csv('inaccuratekb.csv').drop(['Unnamed: 0'],axis=1)
	print (df)
	fig=plt.figure(figsize=(9,4))

	print (df.iloc[1,0:3])
	#l1 = plt.plot(['a','b','c'],df.loc[1,0:2],marker='*',linestyle='-',label='MOMDP(ours)')
	
	#for count,metric in enumerate(list(df)):
	#	pass
		#print metric
	
	ax=plt.subplot(1,2,1)
	xaxis=['Low','Medium','High']
	##############  F1 Score ########################
	l1 = plt.plot(xaxis,df.iloc[0,0:3],marker='*',linestyle='-',label='Learning')
	l2 = plt.plot(xaxis,df.iloc[1,0:3],marker='D',linestyle=':',label='Learning+ Reasoning')
	l3 = plt.plot(xaxis,df.iloc[2,0:3],marker='o',linestyle='--',label='CORPP')
	l4 = plt.plot(xaxis,df.iloc[3,0:3],marker='^',linestyle='-.',label='LeRDeM')
	print df.iloc[0,13:16]
	plt.errorbar(xaxis,df.iloc[0,0:3],yerr=df.iloc[0,13:16], linestyle="None",label=None)
	plt.errorbar(xaxis,df.iloc[1,0:3],yerr=df.iloc[1,13:16], linestyle="None",label=None)
	plt.errorbar(xaxis,df.iloc[2,0:3],yerr=df.iloc[2,13:16], linestyle="None",label=None)
	plt.errorbar(xaxis,df.iloc[3,0:3],yerr=df.iloc[3,13:16], linestyle="None",label=None)

	plt.xlabel('Knowledge level', fontsize= 16)
	plt.ylabel('F1 Score', fontsize= 16)
	ax=plt.subplot(1,2,2)

	##############   Cost  #################
	l1 = plt.plot(xaxis,df.iloc[0,3:6],marker='*',linestyle='-',label='Learning')
	l2 = plt.plot(xaxis,df.iloc[1,3:6],marker='D',linestyle=':',label='Learning+ Reasoning')
	l3 = plt.plot(xaxis,df.iloc[2,3:6],marker='o',linestyle='--',label='CORPP')
	l4 = plt.plot(xaxis,df.iloc[3,3:6],marker='^',linestyle='-.',label='LeRDeM')
	plt.errorbar(xaxis,df.iloc[2,3:6],yerr=df.iloc[2,6:9], linestyle="None",label=None)
	plt.errorbar(xaxis,df.iloc[3,3:6],yerr=df.iloc[3,6:9], linestyle="None",label=None)
	ax.legend(loc='upper left', bbox_to_anchor=(-1.10, 1.15),  shadow=True, ncol=4)
	plt.ylabel('Cost (s)', fontsize= 16)
	plt.xlabel('Knowledge level', fontsize= 16)
	plt.savefig('inaccuratekb.pdf')
	#plt.show()	
   
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
	fig=plt.figure(figsize=(9,3.75))

	print (df.iloc[1,0:3])
	#l1 = plt.plot(['a','b','c'],df.loc[1,0:2],marker='*',linestyle='-',label='MOMDP(ours)')
	
	#for count,metric in enumerate(list(df)):
	#	pass
		#print metric
	
	ax=plt.subplot(1,2,1)
	xaxis=['Partial','Full']
	##############  F1 Score ########################
	l1 = plt.plot(xaxis,df.iloc[0,0:2],marker='*',linestyle='-',label='Learning')
	l2 = plt.plot(xaxis,df.iloc[1,0:2],marker='D',linestyle=':',label='Learning+ Reasoning')

	l4 = plt.plot(xaxis,df.iloc[2,0:2],marker='^',linestyle='-.',label='LeRDeM')

	plt.errorbar(xaxis,df.iloc[0,0:2],yerr=df.iloc[0,9:11], linestyle="None",label=None)
	plt.errorbar(xaxis,df.iloc[1,0:2],yerr=df.iloc[1,9:11], linestyle="None",label=None)
	plt.errorbar(xaxis,df.iloc[2,0:2],yerr=df.iloc[2,9:11], linestyle="None",label=None)
	ax.set_ylim([0.4, 1])

	plt.xlabel('Trajectory', fontsize= 16)
	plt.ylabel('F1 Score', fontsize= 16)

	ax.legend(loc='upper left', bbox_to_anchor=(0.50, 1.15),  shadow=True, ncol=3)
	ax=plt.subplot(1,2,2)

	##############   Cost  #################
	#l1 = plt.plot(xaxis,df.iloc[0,2:4],marker='*',linestyle='-',label='Learning')
	#l2 = plt.plot(xaxis,df.iloc[1,2:4],marker='D',linestyle=':',label='Learning+ Reasoning')
	
	l4 = plt.plot(xaxis,df.iloc[2,2:4],marker='^',linestyle='-.',label='LeRDeM')
	plt.errorbar(xaxis,df.iloc[2,2:4],yerr=df.iloc[2,4:6], linestyle="None",label=None)

	#ax.legend(loc='upper left', bbox_to_anchor=(-1.10, 1.15),  shadow=True, ncol=2)
	plt.ylabel('Cost (s)', fontsize= 16)
	plt.xlabel('Knowledge level', fontsize= 16)

	#plt.show()
	plt.savefig('partialsensor.pdf')	


def main():

	#plotcomparison()
	inaccuratekb()
	#partialsensor()


if __name__ == '__main__':
	main()