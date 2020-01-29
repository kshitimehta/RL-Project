import numpy as np
from rl687.environments.cartpole import Cartpole
from rl687.policies.softmax_theta_phi import SoftmaxThetaPhi

class GenEpCartpole:

    def __init__(self, numStates:int, numActions:int, k: int):
        # self.G = []
        self._numStates = numStates
        self._numActions = numActions
        self._k = k
        
        self.environment = Cartpole()
        self.policy = SoftmaxThetaPhi(numStates,numActions,k)
#        self.policy.k = k

#    @property
#    def batchReturn(self)->str:
#        return self._G
    
    def __call__(self, theta:np.array, numEpisodes:int):
        
        self.policy.parameters = theta
        
        D = []
        
        for episode in range(numEpisodes):
            
            self.environment.reset()
            G_episode = 0
            
            counter = 0
            H = {}
            S = []
            A = []
            R = []	        
            while not self.environment.isEnd:
                state = self.environment.state
                action = self.policy.samplAction(state)
                _, reward, _ = self.environment.step(action)
                
                G_episode += reward
                
                phi_s = self.policy.phiS(state)
                S.append(phi_s)
                A.append(action)
                R.append(reward)

                counter+=1
            
            H['S'] = np.array(S)
            H['A'] = np.array(A)
            H['R'] = np.array(R)
            
            
            D.append(H)
        
        return D

    
    def reset(self):
        self.environment = Cartpole()
        self.policy = SoftmaxThetaPhi(self._numStates,self._numActions)
        self.policy.k = self._k
#        self._G = []

#        self.environment = Cartpole()
#        self.policy = SoftmaxThetaPhi(4,2)
#        self._G = []
