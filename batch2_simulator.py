#!/usr/bin/env python
from parser import Policy,Solver
from pomdp_parser import Model
import numpy as np
from random import randint
import random
#random.seed()
from reason import Reason
from learning import Learning
import pandas as pd
from math import sqrt
import statistics
import sys
from termcolor import colored
import json
import simulator
import csv

class BatchSimulator(simulator.Simulator):


	def __init__(self, pomdpfile='program.pomdp'):

		simulator.Simulator.__init__(self, pomdpfile='program.pomdp' )
		#f= open("newdataset.json","w+")
		#f.close() 

		self.newdatasetfile={}

	def create_instance(self,i,batchnumber):
		random.seed(i)

		person = random.choice(self.identity)
		print (person)
		#raw_input()
		#person = 'visitor'
		#print ('\nIdentity (uniform sampling): [student, visitor, professor] : '), person
#		print ('identity is:'), person
		if person == 'student':
			#place ='classroom'
			place =self.sample(self.location,[0.7,0.3])
			#time ='evening'
			time =self.sample(self.time,[0.15,0.15,0.7])
			intention =self.sample(self.intention,[0.3,0.7])
			#intention = 'not_interested'
		elif person == 'professor':
			place =self.sample(self.location,[0.8,0.2])
			time =self.sample(self.time,[0.7,0.15,0.15])
			intention =self.sample(self.intention,[0.2,0.8])
		else:
			#place = self.sample(self.location,[0.2,0.8])
			place='classroom'
			#time =self.sample(self.time,[0.1,0.7,0.2])
			time = 'afternoon'
			intention =self.sample(self.intention,[0.8,0.2])
			#intention = 'interested'


		ID = self.select_trajectory(intention,batchnumber)
		
		#print (ID)
		#input()
		print ('Sampling time, location and intention for the identity: '+person)
		
		self.instance.append(person)
		self.instance.append(time)    #1
		self.instance.append(place)   #2
		self.instance.append(intention)
		#self.instance.append('label '+str(self.trajectory_label))
		self.instance.append(str(ID[1]))

		print ('Instance: ')
		print (self.instance[0],self.instance[1],self.instance[2], self.instance[3], self.instance[4])

		return self.instance


#	def create_dataset(self, N):

	

#		for i in range(N):
#			self.create_instance(i,1)
#			print (self.instance)
#			if random.choice([True,False]):
#				self.exp_dataset_pos.append(self.instance[4])
#			else:
#				self.exp_dataset_neg.append(self.instance[4])
#			del self.instance[:]

#		print (self.exp_dataset)
		

	def select_trajectory(self, intention,batchnumber):

		#out =self.sample(['experience','dataset'],[(batchnumber-1)/2.0 , 1- (batchnumber-1)/2.0])
		if batchnumber ==1:

			self.trajectory_label = random.choice(['1','0']) 
			ID = self.learning.get_traj(intention)
			

		else:

			if intention=='interested':
				ID = random.choice(self.exp_dataset_pos)
				self.learning.get_traj_from_label(ID)
			else:
				ID = random.choice(self.exp_dataset_neg)
				self.learning.get_traj_from_label(ID)
		

		return ID



	def trial_num(self, num,strategylist,r_thresh,l_thresh, pln_obs_acc,batches):
		#self.learning=Learning('./',xfilename,yfilename)

		df =[0,pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
		
		total_results={}
		subresult=[]
		total_success = {}
		total_cost = {}
		total_tp= {}
		total_tn= {}
		total_fp= {}
		total_fn= {}
		prec={}
		recall={}
		SDcost={}
		reward={}
		SDreward={}

	
		for batchnumber in batches:
			#self.learning=Learning('./',xfilename,yfilename)
			print ('########################## BATCH '+str(batchnumber)+' #########################')
			
			for strategy in strategylist:
				total_success[strategy] = 0
				total_cost[strategy] = 0
				total_tp[strategy]= 0
				total_tn[strategy]= 0
				total_fp[strategy]= 0
				total_fn[strategy]= 0
				prec[strategy] = 0
				recall[strategy] = 0
				reward[strategy]=0
				SDreward[strategy]=[]
				SDcost[strategy]=[] 


				for i in range(num):
					print (colored('######TRIAL:','blue'),colored(i,'red'),colored('#######','blue'))
					subresult=[]
					del self.instance[:] 
					self.create_instance(i,batchnumber)
					time, location =self.observe_fact(i)
			
					for strategy in strategylist:

						c, s, tp, tn, fp, fn, R,value=self.run(strategy,time,location,r_thresh,l_thresh, pln_obs_acc)
						if batchnumber==1:
						#	strategy =='corpp'			
							if (tp ==1 or fp ==1):
								if self.instance[4] not in self.newdatasetfile.keys():
									 self.newdatasetfile[self.instance[4]]='1'
							elif (fn ==1 or tn ==1):
								if self.instance[4] not in self.newdatasetfile.keys(): 
									self.newdatasetfile[self.instance[4]]='0'
							break
						else:
		 
							subresult.append((value, strategy))
							reward[strategy]+=R
							total_cost[strategy]+=c
							total_success[strategy]+=s
							total_tp[strategy]+=tp
							total_tn[strategy]+=tn
							total_fp[strategy]+=fp
							total_fn[strategy]+=fn
							SDcost[strategy].append(c)
							SDreward[strategy].append(R)
					#		print ('total_tp:'), total_tp
					#		print ('total_tn:'), total_tn
					#		print ('total_fp:'), total_fp
					#		print ('total_fn:'), total_fn
							try:
								df[batchnumber].at[strategy,'Reward']= float(reward[strategy])/num
								df[batchnumber].at[strategy,'SDCost']= statistics.stdev(SDcost[strategy])
								df[batchnumber].at[strategy,'SDreward']= statistics.stdev(SDreward[strategy])
								df[batchnumber].at[strategy,'Cost']= float(total_cost[strategy])/num
								df[batchnumber].at[strategy,'Success']= float(total_success[strategy])/num
								prec[strategy] = round(float(total_tp[strategy])/(total_tp[strategy] + total_fp[strategy]),2)
								recall[strategy] = round(float(total_tp[strategy])/(total_tp[strategy] + total_fn[strategy]),2)
								df[batchnumber].at[strategy,'Precision'] = prec[strategy] 
								df[batchnumber].at[strategy,'Recall'] = recall[strategy] 
								df[batchnumber].at[strategy,'F1 Score']= round(2*prec[strategy]*recall[strategy]/(prec[strategy]+recall[strategy]),2)
							
							except:
								print ('Can not divide by zero')
								print ()
								df[batchnumber].at[strategy,'Precision']= 0
								df[batchnumber].at[strategy,'Recall']= 0
								df[batchnumber].at[strategy,'F1 Score']= 0
						
						#self.results[i] =[self.instance[:],subresult[:]]      # important: Do not copy by reference
						#print i
						#print [self.instance,subresult]
						#print self.results

					#with open('data'+str(batchnumber)+'.json', 'w') as f:
					#	json.dump(self.results, f)

						#print self.IDs
						#print len(self.IDs['1'])
						#print len(self.IDs['0'])

					#raw_input()
				#print 'fp',total_fp['learning']
				#print 'fn',total_fn['learning'] 
			
			with open('newdataset.json', 'w+') as fp:
				json.dump(self.newdatasetfile, fp)
			
		return df
		
		

	def print_results(self,df):
		print ('\nWRAP UP OF RESULTS:')
		print (df[1])
		print (df[2])
		print (df[3])

		pd.concat([df[1], df[2],df[3]], axis=0).to_csv('results.csv', sep=',')
		#df[1].to_csv('results.csv', sep=',')
		##df[2].to_csv('results.csv', sep=',')
		#df[3].to_csv('results.csv', sep=',')
		
	


def main():


	batches = [1]
	strategy = ['lstm-corpp']
	print ('startegies are:', strategy)
	Solver()
	a=BatchSimulator()
	
	
	
	r_thresh = 0.5
	num=80
	l_thresh=0.5
	pln_obs_noise = 0.35
	pln_obs_acc =  1- pln_obs_noise

	df = a.trial_num(num,strategy,r_thresh,l_thresh,pln_obs_acc,batches)
	#print (a.exp_dataset_pos)
	#print (a.exp_dataset_neg)
	#print (a.results)
	#with open('data.json', 'w') as f:
	#	json.dump(a.results, f)
	a.print_results(df)
	


if __name__=="__main__":
	main()
