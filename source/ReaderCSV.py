import csv
import itertools
import numpy as np

def parser(filename):
    D = []
    with open(filename, mode = 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        
        line_count = 0
        
        for row in csv_reader:
            if(line_count==0):
                m = row
            if(line_count==1):
                num_actions = row
            if(line_count==2):
                k = row
            if(line_count==3):
                theta_b = row
            if(line_count==4):
                n = row

    
            H ={}
            H.fromkeys(['S','A','R'])
            c = 0
            
            if(line_count>=5 and line_count< int(n[0])+5):
                S = []
                A = []
                R = []

                while(c<len(row)-2):
                    state = float(row[c])
                    phi_s = phiS(np.array(float(state)), int(m[0]), int(k[0]))
                    S.append(phi_s)
                    A.append(int(row[c+1]))
                    R.append(float(row[c+2]))
                    c+=3

                H['S']=np.array(S).squeeze()
                H['A']=np.array(A)
                H['R']=np.array(R)

                D.append(H)


            line_count+=1
    

    m=int(m[0])
    num_actions=int(num_actions[0])
    k=int(k[0])
    n=int(n[0])
    theta_b = np.array([ float(theta_b_i) for theta_b_i in theta_b ])


    return m,num_actions,k,theta_b,n,D

def phiS(state:np.ndarray, numStates:int,k:int)->np.ndarray:
        phi_mat = np.array(list(itertools.product(range(k+1),repeat = numStates)))

        ci_s= phi_mat.dot(state.T)
        phi_s = np.cos(np.pi*ci_s)
        return phi_s


def main():
    parser("data.csv")

if __name__ == '__main__':
    main()

