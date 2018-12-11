#!/usr/bin/env python
from parser import Policy,Solver
from pomdp_parser import Model
import numpy as np
from random import randint
import random
random.seed()
from reason import Reason
from learning import Learning
import pandas as pd
from math import sqrt
import statistics
import sys
from termcolor import colored

class Simulator:
	def __init__(self, pomdpfile='program.pomdp'):
		
		self.time = ['morning','afternoon','evening']
		self.location = ['classroom','library']
		self.identity = ['student','professor','visitor'] 
		self.intention =['interested','not_interested']
		self.reason =Reason('reason_updated.plog')
		self.model = Model(filename='program.pomdp', parsing_print_flag=False)
		self.policy = Policy(5,4 ,output='program.policy')
		self.instance = []
		self.results={}
		self.learning=Learning('./','interposx.csv','interposy.csv')
		self.trajectory_label=0
		

	def sample (self, alist, distribution):

		return np.random.choice(alist, p=distribution)

	def create_instance(self,i):
		random.seed(i)

		person = random.choice(self.identity)
		#person = 'visitor'
		#print ('\nIdentity (uniform sampling): [student, visitor, professor] : '), person
#		print ('identity is:'), person
		if person == 'student':
			place =self.sample(self.location,[0.7,0.3])
			time =self.sample(self.time,[0.4,0.4,0.2])
			intention =self.sample(self.intention,[0.3,0.7])
		elif person == 'professor':
			place =self.sample(self.location,[0.9,0.1])
			time =self.sample(self.time,[0.7,0.2,0.1])
			intention =self.sample(self.intention,[0.1,0.9])
		else:
			place = self.sample(self.location,[0.2,0.8])
			#place='classroom'
			time =self.sample(self.time,[0.1,0.7,0.2])
			#time = 'afternoon'
			intention =self.sample(self.intention,[0.9,0.1])
			#intention = 'interested'


		self.trajectory_label = self.learning.get_traj(intention)

		print ('Sampling time, location and intention for the identity: '+person)
		
		self.instance.append(person)
		self.instance.append(time)    #1
		self.instance.append(place)   #2
		self.instance.append(intention)
		self.instance.append('trajectory with label '+str(self.trajectory_label))
		print ('Instance: ')
		print (self.instance[0],self.instance[1],self.instance[2], self.instance[3],self.instance[4])
		return self.instance


	def observe_fact(self,i):
		random.seed(i)
		#print '\nObservations:'
		time = self.instance[1]
		location = self.instance[2]
		#print ('Observed time: '),time
		#print ('Observed location: '),location
		return time, location

	def init_belief(self, int_prob):
			
		l = len(self.model.states)
		b = np.zeros(l)

		# initialize the beliefs of the states with index=0 evenly
		
		int_prob =float(int_prob)
		init_belief = [0,0,1.0 - int_prob, int_prob, 0]
		b = np.zeros(len(self.model.states))
		for i in range(len(self.model.states)):
			b[i] = init_belief[i]/sum(init_belief)
		print 'The normalized initial belief would be: '
		print b
		return b
			
		return b


	def get_state_index(self,state):

		return self.model.states.index(state)


	def init_state(self):
		state=random.choice(['not_turned_not_interested','not_turned_interested'])
		#print '\nRandomly selected state from [not_forward_not_interested,not_forward_interested] =',state
		s_idx = self.get_state_index(state)
		#print s_idx
		return s_idx, state

	def get_obs_index(self, obs):

		return self.model.observations.index(obs)

	

	def observe(self, a_idx,intention,pln_obs_acc):

		p=pln_obs_acc
		if self.model.actions[a_idx]=='move_forward' and intention=='interested':
			#obs='physical'
			obs=self.sample(['pos','neg'],[p,1-p])
		elif self.model.actions[a_idx]=='move_forward' and intention=='not_interested':
			obs=obs=self.sample(['pos','neg'],[1-p,p])
		elif self.model.actions[a_idx]=='greet' and intention=='interested':
			#obs = 'verbal'
			obs=self.sample(['pos','neg'],[p,1-p])
		elif self.model.actions[a_idx]=='greet' and intention=='not_interested':
			obs=self.sample(['pos','neg'],[1-p,p])
		elif self.model.actions[a_idx]=='turn' and intention=='interested':
			#obs = 'verbal'
			obs=self.sample(['pos','neg'],[p,1-p])
		elif self.model.actions[a_idx]=='turn' and intention=='not_interested':
			obs=self.sample(['pos','neg'],[1-p,p])
		else:
			obs = 'na'
		#l=len(self.model.observations)-1
		#o_idx=randint(0,l)
		o_idx=self.get_obs_index(obs)
		print ('random observation is: ',self.model.observations[o_idx])
		return o_idx

	

	def update(self, a_idx,o_idx,b ):
		b = np.dot(b, self.model.trans_mat[a_idx, :])

		b = [b[i] * self.model.obs_mat[a_idx, i, o_idx] for i in range(len(self.model.states))]
		
		b = b / sum(b)
		return b

	def run(self, strategy,time,location,r_thresh,l_thresh, pln_obs_acc):
		a_cnt=0
		success=0
		tp=0
		tn=0
		fp=0
		fn=0
		cost =0
		R = 0

		if strategy == 'corpp':
			#prob = self.reason.query_nolstm(time, location,'reason_nolstm.plog')
			prob = self.reason.query_nolstm(time, location,'reason_updated.plog')
			print colored('\nSTRATEGY: ','red'),colored(strategy,'red')
			print '\nOur POMDP Model states are: '
			print self.model.states

			s_idx,temp = self.init_state()
			b = self.init_belief(prob)
			
			#print ( 'b shape is,', b.shape )
			#print b

			while True: 
				a_idx=self.policy.select_action(b)
				a = self.model.actions[a_idx]
				a_cnt=a_cnt +1
				if a_cnt>20:
					print ('POLICY IS NOT REPORTING UNDER 20 ACTIONS')
					sys.exit()
				print('action selected',a)
				
				o_idx = self.observe(a_idx,self.instance[3],pln_obs_acc)
				#print ('transition matrix shape is', self.model.trans_mat.shape)
				#print self.model.trans_mat[a_idx,:,:]
				#print ('observation matrix shape is', self.model.obs_mat.shape)
				#print self.model.trans_mat[a_idx,:,:]
				#print s_idx
				R = R + self.model.reward_mat[a_idx,s_idx]
				print 'Reward is : ', cost
				#print ('Total reward is,' , cost)		
				b =self.update(a_idx,o_idx, b)
				print b
				
				

				if 'report' in a:
					if 'not_interested' in a and 'not_interested' == self.instance[3]:
						success= 1
						tn=1
						print 'Trial was successfull'
					elif 'report_interested' in a and 'interested' == self.instance[3]:
						success= 1
						tp=1
						print 'Trial was successful'
					elif 'report_interested' in a and 'not_interested' == self.instance[3]:
						fp=1
						print 'Trial was unsuccessful'
					elif 'not_interested' in a and 'interested' == self.instance[3]:
						fn=1

					print ('Finished\n ')
					
					break
				cost = cost + self.model.reward_mat[a_idx,s_idx]
				print 'cost is : ', cost


		if strategy == 'planning':
			#prob = self.reason.query_nolstm(time, location,'reason_nolstm.plog')
			print colored('\nSTRATEGY','green'),colored(strategy,'green')
			print '\nOur POMDP Model states are: '
			print self.model.states

			s_idx,temp = self.init_state()
			#init_belief = [0.25, 0.25, 0.25, 0.25, 0]
			#b = np.zeros(len(init_belief))
			b = np.ones(len(self.model.states))
			for i in range(len(self.model.states)):
				b[i] = b[i]/len(self.model.states)
			print 'initial belief',b
			while True: 
				a_idx=self.policy.select_action(b)
				a = self.model.actions[a_idx]
			
				print('action selected',a)

				o_idx = self.observe(a_idx,self.instance[3],pln_obs_acc)
				R = R + self.model.reward_mat[a_idx,s_idx]
				print 'R is : ', cost
				#print ('Total reward is,' , cost)		
				b =self.update(a_idx,o_idx, b)
				print b
				
				
				if 'report' in a:
					if 'not_interested' in a and 'not_interested' == self.instance[3]:
						success= 1
						tn=1
						print 'Trial was successfull'
					elif 'report_interested' in a and 'interested' == self.instance[3]:
						success= 1
						tp=1
						print 'Trial was successful'
					elif 'report_interested' in a and 'not_interested' == self.instance[3]:
						fp=1
						print 'Trial was unsuccessful'
					elif 'not_interested' in a and 'interested' == self.instance[3]:
						fn=1

					print ('Finished\n ')
					
					break

				cost = cost + self.model.reward_mat[a_idx,s_idx]
				print 'cost is : ', cost

		if strategy == 'reasoning':
			print colored('\nSTRATEGY is: ','yellow'),colored(strategy,'yellow')
			#prob = self.reason.query_nolstm(time, location,'reason_nolstm.plog')			prob = self.reason.query_nolstm(time, location,'reason_nolstm.plog')
			prob = self.reason.query_nolstm(time, location,'reason_updated.plog')
			
			print ('P(ineterested)| observed time and location: '), prob
			
			prob=float(prob)
			if prob>= r_thresh and 'interested' == self.instance[3] :
				success = 1
				print ('Greater than threshold ='+str(r_thresh)+', -> human IS interested')
				print 'Trial was successful'
				tp=1
			elif prob>= r_thresh and 'not_interested' == self.instance[3]:
				success=0
				fp=1
				print ('Greater than threshold ='+str(r_thresh)+', -> human IS interested')
				print 'Trial was unsuccessful'
			elif prob < r_thresh and 'interested' == self.instance[3] :
				success = 0
				print ('Less than threshold ='+str(r_thresh)+', -> human IS NOT interested')
				fn=1
				print 'Trial was unsuccessful'
			else:
				success = 1
				print ('Less than threshold ='+str(r_thresh)+', -> human IS NOT interested')
				print 'Trial was successful'
				tn=1

		if strategy=='learning':
			
			print colored('\nStrategy is: ','blue'),colored(strategy,'blue')
			res = self.learning.predict()
		
			if res>l_thresh and self.trajectory_label ==1.0:
				print ('CASE I the trajectory shows person is interested')
				success=1
				tp=1
			elif res<l_thresh and self.trajectory_label ==0:
				
				print ('CASE II the person is not interested')
				success =1
				tn=1
			elif res>l_thresh and self.trajectory_label ==0:
				sucess=0
				fp =1
				print ('CASE III the trajectory shows person is interested')
			elif res <l_thresh and self.trajectory_label == 1.0:
				fn =1
				success =0
				('CASE IV the person is not interested')
		


		if strategy =='lstm-corpp':
			print colored('\nSTRATEGY is: ','magenta'),colored(strategy,'magenta')
			res = self.learning.predict()
			#do not change 0.2 below
			if res > l_thresh:
				#prob = self.reason.query(time, location,'one','reason.plog')
				prob = self.reason.query(time, location,'one','reason_updated.plog')
			else:
				prob = self.reason.query(time, location,'zero','reason_updated.plog')
			print '\nOur POMDP Model states are: '
			print self.model.states
			
			s_idx,temp = self.init_state()
			b = self.init_belief(prob)

			while True: 
				a_idx=self.policy.select_action(b)
				a = self.model.actions[a_idx]
			
				print('action selected',a)

				o_idx = self.observe(a_idx,self.instance[3],pln_obs_acc)
				R = R + self.model.reward_mat[a_idx,s_idx]
				print 'R is : ', cost
				#print ('Total reward is,' , cost)		
				b =self.update(a_idx,o_idx, b)
				print b
				
				
				if 'report' in a:
					if 'not_interested' in a and 'not_interested' == self.instance[3]:
						success= 1
						tn=1
						print 'Trial was successfull'
					elif 'report_interested' in a and 'interested' == self.instance[3]:
						success= 1
						tp=1
						print 'Trial was successful'
					elif 'report_interested' in a and 'not_interested' == self.instance[3]:
						fp=1
						print 'Trial was unsuccessful'
					elif 'not_interested' in a and 'interested' == self.instance[3]:
						fn=1

					print ('Finished\n ')
					
					break
				cost = cost + self.model.reward_mat[a_idx,s_idx]
				print 'cost is : ', cost

		if strategy =='lreasoning':
			print colored('\nStrategy is: learning + reasoning ','cyan')
			res = self.learning.predict()
			if res > l_thresh:
				prob = self.reason.query(time, location,'one','reason_updated.plog')
			else:
				prob = self.reason.query(time, location,'zero','reason_updated.plog')
			lr_thresh = r_thresh
			prob=float(prob)
			if prob>= r_thresh and 'interested' == self.instance[3] :
				success = 1
				print ('Greater than threshold ='+str(r_thresh)+', -> human IS interested')
				print 'Trial was successful'
				tp=1
			elif prob>= r_thresh and 'not_interested' == self.instance[3]:
				success=0
				fp=1
				print ('Greater than threshold ='+str(r_thresh)+', -> human IS interested')
				print 'Trial was unsuccessful'
			elif prob < r_thresh and 'interested' == self.instance[3] :
				success = 0
				print ('Less than threshold ='+str(r_thresh)+', -> human IS NOT interested')
				fn=1
				print 'Trial was unsuccessful'
			else:
				success = 1
				print ('Less than threshold ='+str(r_thresh)+', -> human IS NOT interested')
				print 'Trial was successful'
				tn=1

						
		return cost, success, tp, tn, fp, fn,R


	def trial_num(self, num,strategylist,r_thresh,l_thresh, pln_obs_acc):
		df = pd.DataFrame()

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

			del self.instance[:] 
			self.create_instance(i)
			time, location =self.observe_fact(i)
		
			for strategy in strategylist:	
			
				c, s, tp, tn, fp, fn, R=self.run(strategy,time,location,r_thresh,l_thresh, pln_obs_acc)
				
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
					df.at[strategy,'Reward']= float(reward[strategy])/num
					df.at[strategy,'SDCost']= statistics.stdev(SDcost[strategy])
					df.at[strategy,'SDreward']= statistics.stdev(SDreward[strategy])
					df.at[strategy,'Cost']= float(total_cost[strategy])/num
					df.at[strategy,'Success']= float(total_success[strategy])/num
					prec[strategy] = round(float(total_tp[strategy])/(total_tp[strategy] + total_fp[strategy]),2)
					recall[strategy] = round(float(total_tp[strategy])/(total_tp[strategy] + total_fn[strategy]),2)
					df.at[strategy,'Precision'] = prec[strategy] 
					df.at[strategy,'Recall'] = recall[strategy] 
					df.at[strategy,'F1 Score']= round(2*prec[strategy]*recall[strategy]/(prec[strategy]+recall[strategy]),2)
				
				except:
					print 'Can not divide by zero'
					df.at[strategy,'Precision']= 0
					df.at[strategy,'Recall']= 0
					df.at[strategy,'F1 Score']= 0
		
		
		#print 'fp',total_fp['learning']
		#print 'fn',total_fn['learning'] 
	
		#self.instance =[]
		return df
		
		

	def print_results(self,df):
		print '\nWRAP UP OF RESULTS:'
		print df
	


def main():
	strategy = ['learning','lreasoning', 'reasoning','planning','corpp','lstm-corpp']
	#strategy = ['reasoning','learning','lreasoning']
	print 'startegies are:', strategy
	Solver()
	a=Simulator()
	r_thresh = 0.5
	num=5000
	l_thresh=0.2
	pln_obs_noise = 0.25
	pln_obs_acc =  1- pln_obs_noise
	df = a.trial_num(num,strategy,r_thresh,l_thresh,pln_obs_acc)
	a.print_results(df)



if __name__=="__main__":
	main()
