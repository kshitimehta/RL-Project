import numpy as np

#PDIS_H calculates the fraction value of PDIS for each H_i

def PDIS_H(H,pi_e,pi_b,gamma):
    L,_ = H['S'].shape
    value = 0
   
    for t in range(L):
        S = H["S"]
        A = H["A"]
        R = H["R"]
        R_t = R[t]
        gamma**t
        product = 1
        
        for j in range(t+1):
            frac = None
            S_j = S[j]
            A_j = A[j]
            piE= pi_e(S_j,A_j)
            piB= pi_b(S_j,A_j)
            if piB!=0:
                frac = piE/piB
            else:
                frac = 0
            product = product * frac

        value = value + (gamma**t)*product*R_t
    return value 

#PDIS_D calculates the average value of PDIS for all H_i

def PDIS_D(D,pi_e,pi_b,gamma):
    n = len(D)
    pdis_h = []
    
    for i in range(n):
        H = D[i]
        pdis_h.append(PDIS_H(H,pi_e,pi_b,gamma)) 

    pdis_h = np.array(pdis_h)
    pdis_d = np.average(pdis_h)
    
    return pdis_h,pdis_d
