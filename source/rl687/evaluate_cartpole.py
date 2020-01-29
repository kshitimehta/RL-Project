import numpy as np
from rl687.environments.cartpole import Cartpole
from rl687.policies.softmax_theta_phi import SoftmaxThetaPhi
class EvaluateCartpole:
    def __init__(self):
        self.environment = Cartpole()
        self.policy = SoftmaxThetaPhi(4,2)
        self._G = []
    
    @property
    def batchReturn(self)->str:
        return self._G
    
    def __call__(self, theta:np.array, numEpisodes:int):

        self.policy.parameters = theta

        for episode in range(numEpisodes):

            self.environment.reset()
            G_episode = 0
            
            counter = 0
	        
            while not self.environment.isEnd:

                state = self.environment.state
                action = self.policy.samplAction(state)
                _, reward, _ = self.environment.step(action)
                
                G_episode += reward
                
                counter+=1
                
            self._G.append(G_episode)

        return np.mean(self._G)
    
    def reset(self):
        self.environment = Cartpole()
        self.policy = SoftmaxThetaPhi(4,2)
        self._G = []
