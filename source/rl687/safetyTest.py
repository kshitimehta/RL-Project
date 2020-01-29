from rl687.policies.softmax_theta_phi import SoftmaxThetaPhi
from rl687.pdis import PDIS_D
from scipy import stats
import sys
import numpy as np

def safetyTest(D_s,theta_b, theta_e, policy_b, policy_e, delta:int, c:int,gamma:int):
    policy_e.parameters = theta_e
    policy_b.parameters = theta_b
    D_s_size = len(D_s)
    pdis_d_arr,pdis_d = PDIS_D(D_s,policy_e.piPhiS, policy_b.piPhiS, gamma)
    sigma_c = np.std(pdis_d_arr,ddof=1)
    safety_term = (sigma_c/np.sqrt(D_s_size))*stats.t.ppf(1-delta,D_s_size-1)
    
    Val = pdis_d - safety_term
    
    if(Val>c):
        return True
    else:
        return False    

