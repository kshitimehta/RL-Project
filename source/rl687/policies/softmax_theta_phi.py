import numpy as np
from .skeleton import Policy
from typing import Union


from itertools import product

class SoftmaxThetaPhi(Policy):
    """
    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numStates:int, numActions: int, k:int):
        self._numActions = numActions
        self._numStates = numStates
        self._k = k
        self._parameters = np.zeros((numActions,(self._k+1)**numStates))
        self._phi = np.array(list(product(range(self._k+1),repeat = self._numStates)))

    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._parameters = p.reshape(self._parameters.shape)

    @property
    def k(self)->int:
        return self._k
        
    @k.setter
    def k(self, k_value:int):
        self._k = k_value

    def __call__(self, state:np.ndarray, action=None)->Union[float, np.ndarray]:
        #TODO
        action_prob = self.getActionProbabilities(state)
              
        if(action == None):
            return action_prob
        else:
            return action_prob[action]
        # pass

    def samplAction(self, state:np.ndarray)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        action = np.random.choice (self._numActions, p = self.getActionProbabilities(state))
        
        return action
        #TODO
        # pass

    def getActionProbabilities(self, state:np.ndarray)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """
        ci_s= self._phi.dot(state.T)
        phi_s = np.cos(np.pi*ci_s)   
        param_a_phi = self._parameters.dot(phi_s.T)
        expnent = np.exp(param_a_phi-max(param_a_phi))
        action_prob = expnent / expnent.sum(axis=0)
                
        return action_prob

    def pi(self, state:np.ndarray, action)->np.ndarray:
        action_prob = self.getActionProbabilities(state)
        return action_prob[action]
    
    def piArr(self, phiSArray:np.ndarray, actionArr:np.ndarray)->np.ndarray: 
        n = actionArr.size
        
        pi_b_list = []
        for episode in n:    
            action = actionArr[episode]
            phi_s = phiSArray[episode]
            param_a_phi = self._parameters.dot(phi_s.T)
            action_prob = np.exp(param_a_phi)/np.sum(np.exp(param_a_phi))
            pi_b_list.append(action_prob[action])
        
        pi_b = pi_b_list.array(pi_b_list)
        return pi_b

    def phiS(self, state:np.ndarray)->np.ndarray:
        ci_s= self._phi.dot(state.T)
        phi_s = np.cos(np.pi*ci_s)
        return phi_s
    
    def getActionProbabilitiesPhiS(self, phi_s:np.ndarray)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """

        param_a_phi = self._parameters.dot(phi_s.T)
        expnent = np.exp(param_a_phi-max(param_a_phi))
        action_prob = expnent / expnent.sum(axis=0)
        return action_prob
    
    def piPhiS(self, phi_s:np.ndarray, action)->np.ndarray:
        action_prob = self.getActionProbabilitiesPhiS(phi_s)
        return action_prob[action]
