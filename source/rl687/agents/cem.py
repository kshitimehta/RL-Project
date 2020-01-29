import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, numEpisodes:int, evaluationFunction:Callable, epsilon:float=0.0001):

        self._name = "Cross_Entropy_Method"
        
        self._theta = theta
        # self._parameters = theta
        self._sigma = sigma
        self._popSize = popSize
        self._numElite = numElite 
        self._numEpisodes = numEpisodes 
        self._evaluationFunction = evaluationFunction 
        self._epsilon = epsilon
               
        # self._theta = None #TODO: set this value to the current mean parameter vector
        self._Sigma = self._sigma*np.identity((len(self._theta))) #TODO: set this value to the current covariance matrix
        
        self._orig_theta = theta
        self._orig_Sigma = self._sigma*np.identity((len(self._theta))) 
        
        #TODO
        # pass
        

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._theta
        # pass


    def train(self)->np.ndarray:
        #TODO

        theta_arr = np.zeros((self._popSize,len(self._theta)))
        J_hat_arr = np.zeros(self._popSize)
        
        self._evaluationFunction.reset()
        
        for k in range(self._popSize):
            theta_arr[k] = np.random.multivariate_normal(self._theta, self._Sigma)
            J_hat_arr[k] = self._evaluationFunction(theta_arr[k], self._numEpisodes)
#            pass
                    
        eliteIdx = J_hat_arr.argsort()[::-1][:self._numElite]
#        print("eliteIdx.shape: ",eliteIdx.shape)
#        print("eliteIdx: ",eliteIdx)

        self._theta = np.average(theta_arr[eliteIdx],axis = 0)
#        self._theta = np.sum(theta_arr[eliteIdx],axis = 0)/(self._numElite)
#        print("self._theta",self._theta)
    
        eliteThetaZeroMean = np.expand_dims((theta_arr[eliteIdx] - self._theta),axis = -1)
#        print("eliteThetaZeroMean.shape: ",eliteThetaZeroMean.shape)
        
        covMatrix = (np.matmul(eliteThetaZeroMean,eliteThetaZeroMean.transpose(0,2,1)))
#        print("covMatrix.shape: ",covMatrix.shape)
#        
        covMatrix = np.sum(covMatrix,axis = 0)
#        print("covMatrix.shape: ",covMatrix.shape)
#        
        self._Sigma = (self._epsilon*np.identity(len(self._theta)) + covMatrix)/(self._numElite + self._epsilon)
#        print("self._Sigma.shape: ",self._Sigma.shape)
#        
        return self._theta
        # pass

    def reset(self)->None:
        #TODO
        self._theta = self._orig_theta
        self._Sigma = self._orig_Sigma
        # pass
