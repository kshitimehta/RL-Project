import numpy as np
from .bbo_agent import BBOAgent

# from rl687.policies.tabular_softmax import TabularSoftmax

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """
    
    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, numEpisodes:int=10):
        self._name = "First_Choice_Hill_Climbing"
        #TODO
        self._orig_parameters = theta
        self._parameters = theta
        self._sigma = sigma
        self._evaluationFunction = evaluationFunction
        self._numEpisodes = numEpisodes
        self.J_hat = self._evaluationFunction(self._parameters,self._numEpisodes)
        
        # self._name = "First_Choice_Hill_Climbing"
        # self.parameters = theta
        # self.sigma = sigma
        # self.evaluationFunction = evaluationFunction
        # self.numEpisodes = numEpisodes
        # self.J_hat = self.evaluationFunction(self.parameters,self.numEpisodes)
        # pass

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._parameters
        pass

    def train(self)->np.ndarray:
        #TODO
        
        # parameters_prime = np.random.multivariate_normal(self._parameters, self._sigma*np.identity(self._parameters.shape))        
        parameters_prime = np.random.multivariate_normal(self._parameters, self._sigma*np.identity(len(self._parameters)))
        
        self._evaluationFunction.reset()
        
        J_hat_prime = self._evaluationFunction(parameters_prime,self._numEpisodes)
        
        if (J_hat_prime>self.J_hat):
            self._parameters = parameters_prime
            self.J_hat = J_hat_prime
        
        return self._parameters
        pass

    def reset(self)->None:
        #TODO
        self._parameters = self._orig_parameters
        # self._sigma = 1
        # self._evaluationFunction = evaluationFunction
        # self._numEpisodes = numEpisodes
        self.J_hat = self._evaluationFunction(self._parameters,self._numEpisodes)
        
        pass
