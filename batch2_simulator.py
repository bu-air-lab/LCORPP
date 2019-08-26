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


	def create_instance(self,i,batchnumber):
		random.seed(i)

		person = random.choice(self.identity)
		print person
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


		ID = self.select_trajectory(intention, batchnumber)
		

		print ('Sampling time, location and intention for the identity: '+person)
		
		self.instance.append(person)
		self.instance.append(time)    #1
		self.instance.append(place)   #2
		self.instance.append(intention)
		#self.instance.append('label '+str(self.trajectory_label))
		self.instance.append(str(ID))
		print ('Instance: ')
		print (self.instance[0],self.instance[1],self.instance[2], self.instance[3], self.instance[4])

		return self.instance


	def create_dataset(self, N):

		for i in range(N):
			self.create_instance(i,1)
			print self.instance
			self.exp_dataset[self.instance[4]]=random.choice(['1','0'])
			del self.instance[:]

		print (self.exp_dataset)


	def select_trajectory(self, intention, batchnumber):

		#out =self.sample(['experience','dataset'],[(batchnumber-1)/2.0 , 1- (batchnumber-1)/2.0])
		if batchnumber==1:
			out='dataset'
		elif batchnumber==3:
			out= 'experience'
		else:
			out = self.sample(['experience','dataset'],[0.8,0.2])
		if out=='experience':

			if intention=='interested':
				ID = random.choice(self.IDs['1'])
				self.learning.get_traj_from_label(ID)
			else:
				ID = random.choice(self.IDs['0'])
				self.learning.get_traj_from_label(ID)
		else:

			self.trajectory_label, ID = self.learning.get_traj(intention)

		return ID

	def trial_num(self, num,strategylist,r_thresh,l_thresh, pln_obs_acc,batches):
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
			print '########################## BATCH '+str(batchnumber)+' #########################'

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
				print colored('######TRIAL:','blue'),colored(i,'red'),colored('#######','blue')
				subresult=[]
				del self.instance[:] 
				self.create_instance(i,batchnumber)
				time, location =self.observe_fact(i)
		
				for strategy in strategylist:

					#if batchnumber==1:
					#	strategy =='corpp'			
					c, s, tp, tn, fp, fn, R,value=self.run(strategy,time,location,r_thresh,l_thresh, pln_obs_acc)
					if (tp ==1 or fp ==1):
						if self.instance[5] not in self.IDs['1']:
							self.IDs['1'].append(self.instance[5]) 
					elif (fn ==1 or tn ==1):
						if self.instance[5] not in self.IDs['0']: 
							self.IDs['0'].append(self.instance[5]) 
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
						print 'Can not divide by zero'
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
	
		
			
		return df
		
		

	def print_results(self,df):
		print '\nWRAP UP OF RESULTS:'
		print df[1]
		print df[2]
		print df[3]

		pd.concat([df[1], df[2],df[3]], axis=0).to_csv('results.csv', sep=',')
		#df[1].to_csv('results.csv', sep=',')
		##df[2].to_csv('results.csv', sep=',')
		#df[3].to_csv('results.csv', sep=',')
		
	


def main():

	batches = [1,2,3]
	strategy = ['lstm-corpp']
	print 'startegies are:', strategy
	Solver()
	a=BatchSimulator()
	a.create_dataset(10)

	'''
	r_thresh = 0.5
	num=200
	l_thresh=0.5
	pln_obs_noise = 0.35
	pln_obs_acc =  1- pln_obs_noise
	df = a.trial_num(num,strategy,r_thresh,l_thresh,pln_obs_acc,batches)

	print a.results
	#with open('data.json', 'w') as f:
	#	json.dump(a.results, f)
	a.print_results(df)
	'''


if __name__=="__main__":
	main()
