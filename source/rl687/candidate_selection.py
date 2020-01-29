"""
Created on Fri Dec  6 12:15:54 2019

@author: Kshiti
"""

import numpy as np
import sys
import cma
from rl687.policies.softmax_theta_phi import SoftmaxThetaPhi
from rl687.pdis import PDIS_D
from scipy import stats
import multiprocessing as mp

class CandidateSelection:
    
    def __init__(self, D_c, D_s, theta_b, policy_b, policy_e, delta:int, c:int, gamma:int , maxIter:int=30):
        self._D_c = D_c
        self._D_s = D_s
        self._D_s_size = len(D_s)
        j = self.jPiB()
        print("Baseline return: ", j)
        self._c = c*j
        self._delta = delta
        self._theta_b = theta_b
        
        self._policy_b = policy_b
        self._policy_e = policy_e
        
        self._policy_b.parameters = self._theta_b
        self._gamma = gamma
        self._maxIter = maxIter
        
    @property
    def theta_b(self)->np.ndarray:
        return self._theta_b.flatten()
    
    @theta_b.setter
    def theta_b(self, p:np.ndarray):
        self._theta_b = p.reshape(self._theta_b.shape)

    @staticmethod
    def parallelization(function,i,n_processes = 8):
        with mp.Pool(processes=n_processes) as pool:
            return pool.map(function,i)

    def evaluateCMAES(self,sigma):
#        cma_es = cma.CMAEvolutionStrategy(theta_b, sigma)
#        print(self._theta_b.shape)
#        print(self._theta_b)
         cmaEs = cma.CMAEvolutionStrategy(self.theta_b,sigma,{'maxiter':self._maxIter,'popsize':8})
         #pool = mp.Pool(cmaEs.popsize)
         print(cmaEs.popsize)
         mp.freeze_support()         
         while not cmaEs.stop():
             X = cmaEs.ask()
             f_values = self.parallelization(self.BarrierFunction,X,len(X))
             # use chunksize parameter as es.popsize/len(pool)?
             cmaEs.tell(X, f_values)
             cmaEs.disp()
             cmaEs.logger.add()
             print(cmaEs.result[1])
             print(cmaEs.result[0])

        #cmaEs.optimize(self.BarrierFunction)
        #cmaEs.disp()
        #print(cmaEs.result[1])
         theta_c = cmaEs.result[0]
         jPiC = cmaEs.result[1]
         print("optimal policy result: ",-jPiC)
         
         return theta_c
        
    def BarrierFunction(self, theta_e):
        self._policy_e.parameters = theta_e
        degree = self._D_s_size-1
#        self._policy_b.parameters = self._theta_b
        pdis_d_arr,pdis_d = PDIS_D(self._D_c,self._policy_e.piPhiS, self._policy_b.piPhiS, self._gamma)
        std_dev = np.std(pdis_d_arr,ddof=1)
        safety_term = 2*(std_dev/np.sqrt(self._D_s_size))*stats.t.ppf(1-self._delta,degree)
        
#        print("safety_term: ",safety_term)
        
        barrierFunctionVal = pdis_d - safety_term
        
        if(barrierFunctionVal > self._c):
            print(pdis_d);
            return -pdis_d
        else:
            return 100000    
        
    def jPiB(self):
        numEpisodes = len(self._D_c)
        R = []
        for ep in range(numEpisodes):
            H = self._D_c[ep]
            R+=(H['R']).tolist() 
#        print(R)
        R_arr = np.array(R)
#        print(R_arr.shape)
        jPiB_return = np.average(R_arr)
        return jPiB_return
