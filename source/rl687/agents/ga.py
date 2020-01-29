import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, 
                 initPopulationFunction:Callable, numElite:int=1, numEpisodes:int=10):
        self._name = "Genetic_Algorithm"
        self._population = initPopulationFunction(populationSize) #TODO: set this value to the most recently created generation
#        self._population = #TODO: set this value to the most recently created generation
        
        self._evaluationFunction = evaluationFunction
        self._initPopulationFunction = initPopulationFunction
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._populationSize = populationSize
        
        self._alpha = 2.5
#        self._numParents = numElit
        self._numParents = 3
        
        self._theta_arr = self._population
        self._theta = np.zeros(self._theta_arr.shape[-1])
        
        self._orig_theta_arr = self._theta_arr
        self._orig_theta = self._theta
        self._orig_population = self._population
        #TODO 
#        pass

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._theta
#        pass

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        #TODO
#        child = np.zeros()
#        child = np.random.multivariate_normal(parent,self._alpha*np.identity(len(parent)))
#        return child
        child = np.zeros(len(parent))
        child = parent + self._alpha*np.random.normal(child,1)
        return child
#        pass

    def train(self)->np.ndarray:
        #TODO
        self._evaluationFunction.reset()

        J_hat_arr = np.zeros(self._populationSize)
        for k in range(self._populationSize):
            J_hat_arr[k] = self._evaluationFunction(self._theta_arr[k],self._numEpisodes)
#            
#        print("J_hat_arr.shape: ",J_hat_arr.shape)
        
        parentIdx = J_hat_arr.argsort()[::-1][:self._numParents]
        eliteIdx = J_hat_arr.argsort()[::-1][:self._numElite]
        
#        print("parentIdx.shape :",parentIdx.shape)
#        print("eliteIdx.shape :",eliteIdx.shape)
        
        thetaParents = self._theta_arr[parentIdx]
        thetaElite = self._theta_arr[eliteIdx]
#        print("thetaParents.shape :",thetaParents.shape)
#        print("thetaElite.shape :",thetaElite.shape)
        
        childrenIdx = np.random.choice(self._numParents,size = (self._populationSize - self._numElite), replace = True)
#        print("childrenIdx.shape :",childrenIdx.shape)
        
#        print(childrenIdx)
#        
#        vecMutate = np.vectorize(self._mutate)        
#        thetaChildren = vecMutate(thetaParents[childrenIdx])
#        
        thetaChildren = np.zeros((len(childrenIdx),len(self._theta)))
        for i in range(len(childrenIdx)):
            thetaChildren[i] = self._mutate(thetaParents[childrenIdx[i]])
            
#        thetaChildren = []
#        for i in range(len(childrenIdx)):    
#            thetaChildren.append(self._mutate(thetaParents[childrenIdx[i]])) 
#        
#        thetaChildren = np.array(thetaChildren)
#        print("thetaChildren.shape :",thetaChildren.shape)
        
        self._theta_arr[:self._numElite] = thetaElite
#        print("self._theta_arr[:self._numElite].shape :",self._theta_arr[:self._numElite].shape)
        self._theta_arr[self._numElite:] = thetaChildren
#        print("self._theta_arr.shape :",self._theta_arr.shape)
#        self._theta = self._theta        
        self._theta = self._theta_arr[0]

#        print("self._theta.shape :",self._theta.shape)
#        
        return self._theta
#        pass

    def reset(self)->None:
        #TODO
        self._theta = self._orig_theta
        self.theta_arr = self._orig_theta_arr        
        self._population = self._orig_population 

#        pass