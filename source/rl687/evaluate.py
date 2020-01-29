import numpy as np
from rl687.environments.gridworld import Gridworld
from rl687.policies.tabular_softmax import TabularSoftmax

class Evaluate:
    def __init__(self):
        self.environment = Gridworld()
        self.policy = TabularSoftmax(25,4)
        self._G = []
    
    @property
    def batchReturn(self)->str:
        return self._G
    
    def __call__(self, theta:np.array, numEpisodes:int):
#	    print("Evaluating Gridworld")
#        self._G = [] #reset G at every call
	    # environment = Gridworld()
        # policy = TabularSoftmax(25,4)
        self.policy.parameters = theta
#        print("numEpisodes",numEpisodes)
        
        Count = 200
        
        for episode in range(numEpisodes):
	        
            self.environment.reset()
            G_episode = 0
            
            counter = 0
            ctr=0
            while not self.environment.isEnd:

                if(counter>=Count):
                    G_episode = -50
                    break
                state = self.environment.state
                action = self.policy.samplAction(state)
                _, reward, _ = self.environment.step(action)
                
                G_episode += (self.environment.gamma**ctr)*reward
#                G_episode += reward
                
                counter+=1
                ctr+=1
	        # self.returns.append(Gi)
            self._G.append(G_episode)
#            if (episode % 50 == 0):
#                print(G_episode)

	    # print("Mean Return ", np.mean(G))
        return np.mean(self._G)
    
    def reset(self):
        self.environment = Gridworld()
        self.policy = TabularSoftmax(25,4)
        self._G = []
