from rl687.policies.softmax_theta_phi import SoftmaxThetaPhi
from rl687.pdis import PDIS_D
from rl687.safetyTest import safetyTest
from rl687.GenerateData import GenEpCartpole
import cma
from itertools import product
from rl687.candidate_selection import CandidateSelection
from ReaderCSV import parser
from rl687.environments.cartpole import Cartpole
import numpy as np

from datetime import datetime


np.random.seed(100)


def problem1():
    
########## Episode Generation Cartpole #########################
#    k = 3
#    numActions = 2
##    m
##    numStates = 4
#    m = 4
#    
#    GenerateData = GenEpCartpole(m,numActions,k)
#    
#    theta_b = np.ones((numActions,(k+1)**m))
#
##    numEpisodes = 1001
##    D = gen_ep_cartpole(theta_b,numEpisodes)


    m,numActions,k,theta_b,n,D = parser("data.csv")
#    print(theta_b)
    np.savetxt("theta_b.csv", theta_b, delimiter=',')
    i = 1

    while(i<=100):
        
        numEpisodes = len(D)
        print(numEpisodes)
        #train:test -> 60:40 split
        np.random.shuffle(D)
        D_c = D[:int(numEpisodes*0.6)]
        D_s = D[int(numEpisodes*0.6+1):]
        
        policy_e = SoftmaxThetaPhi(m,numActions,k)
        policy_b = SoftmaxThetaPhi(m,numActions,k)

        delta = 0.1
        c = 1.5
        gamma = 1
        maxIter = 1   
        candidate_selection = CandidateSelection(D_c,D_s,theta_b,policy_b,policy_e,delta,c,gamma,maxIter)

        sigma = 2.0*(np.dot(theta_b.T,theta_b) + 1.0)
        print("sigma: ",sigma)
    #    np.array([1.18E+01, 2.41E+00, -6.95E+00, -2.54E+01])
        theta_c = candidate_selection.evaluateCMAES(sigma)
        policy_c = SoftmaxThetaPhi(m,numActions,k)
        if (safetyTest(D_s, theta_b, theta_c, policy_b, policy_c,delta,c,gamma)):
            with open(f"{i}.csv","w") as f:
                f.write(','.join([str(value) for value in theta_c.tolist()]))
                #np.savetxt(f"{i}.csv", theta_c, delimiter=',')
            i+=1
        else:
            print("NSF")
    
def main():
    problem1()


if __name__ == "__main__":
    main()
