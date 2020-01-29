from rl687.evaluate import Evaluate
from rl687.evaluate_cartpole import EvaluateCartpole
from rl687.agents.cem import CEM
from rl687.agents.fchc import FCHC
from rl687.agents.ga import GA

import matplotlib.pyplot as plt

from rl687.environments.cartpole import Cartpole
#from rl687 import evaluate 


import numpy as np

# import utils
from datetime import datetime
# datetime object containing current date and time

np.random.seed(333)

def problem1():
    """
    Apply the CEM algorithm to the More-Watery 687-Gridworld. Use a tabular 
    softmax policy. Search the space of hyperparameters for hyperparameters 
    that work well. Report how you searched the hyperparameters, 
    what hyperparameters you found worked best, and present a learning curve
    plot using these hyperparameters, as described in class. This plot may be 
    over any number of episodes, but should show convergence to a nearly 
    optimal policy. The plot should average over at least 500 trials and 
    should include standard error or standard deviation error bars. Say which 
    error bar variant you used. 
    """
    print("cem-gridworld-tabular_softmax")
    
    theta = np.zeros(100) 
    sigma = 1
    popSize = 10
    numElite = 3
    numEpisodes = 10
    evaluate = Evaluate()
    epsilon = 5
    
    cem = CEM(theta,sigma, popSize, numElite, numEpisodes,evaluate,epsilon)
    
#    numTrials = 50
    numTrials = 50
    numIterations = 250
#    numIterations = 50
#    numIterations = 20
    total_episodes = numIterations*numEpisodes*popSize # 20*50*10
    
    results = np.zeros((numTrials,total_episodes))
    
    for trial in range(numTrials):
        cem.reset()
        for i in range(numIterations):
            #DEBUG
            if (i%5 == 0):
                print("cem: ","trial: ",trial,"/",numTrials," iteration: ",i,"/",numIterations)
            cem.train()
            
            batch_start = (i*numEpisodes)*popSize
            batch_end = ((i+1)*numEpisodes)*popSize
            
            results[trial,batch_start:batch_end] = np.array(evaluate.batchReturn)
    
    average_results = np.average(np.array(results), axis=0)
    std_results = np.std(np.array(results), axis=0)
    maximumEpisodes = average_results.shape[0]
    max_avg = np.max(average_results)
    
    plt.errorbar(np.array([i for i in range(maximumEpisodes)]), average_results, std_results,fmt = 'o', marker='.', ecolor='aqua')
    plt.grid(True)
    plt.axhline(max_avg)
    plt.text(0, max_avg, "max: "+str(round(max_avg,2)),fontsize=15, backgroundcolor='w')
    
    plt_name = "cem_gridworld"

    now = datetime.now()
    param_string = "_numTrials_"+str(numTrials)+"_numIter_" \
        + str(numIterations) + "_popSize_" +str(popSize)
    dt_string = now.strftime("_t_%H_%M")
    
    plt_name+= param_string
    plt_name+= dt_string
    print("plt_name=", plt_name)
    plt_path = "images/"+plt_name + ".png"

#    plot_min = -100
#    plot_max = 10
#    plt.ylim(average_results.min(), average_results.max())
#    plt.ylim(plot_min, plot_max)
    
    plt.savefig(plt_path,dpi=200)
    plt.show()

    np.save("data/"+"results_"+plt_name, results)
    np.save("data/"+"average_results_"+plt_name, average_results)
    np.save("data/"+"std_results_"+plt_name, std_results)

def problem2():
    """
    Repeat the previous question, but using first-choice hill-climbing on the 
    More-Watery 687-Gridworld domain. Report the same quantities.
    """
    
    print("fchc-gridworld-tabular_softmax")

    theta = np.zeros(100) 
    sigma = 1
    evaluate = Evaluate()
    num_episodes = 10
#    num_episodes = 50
    
    fchc = FCHC(theta, sigma, evaluate, num_episodes) # init fchc (calculates J_hat)    
    
#    numTrials = 50
    numTrials = 50
#    numIterations = 100
    numIterations = 2500
    total_episodes = numIterations*num_episodes # 100*50
    
    results = np.zeros((numTrials,total_episodes))
    
    for trial in range(numTrials):
        fchc.reset()
        for i in range(numIterations):
            #DEBUG
            if (i%5 == 0):
                print("fchc: ","trial: ",trial,"/",numTrials," iteration: ",i,"/",numIterations)
            fchc.train()
            
            batch_start = (i*num_episodes)
            batch_end = ((i+1)*num_episodes)
            
            results[trial,batch_start:batch_end] = np.array(evaluate.batchReturn)

    average_results = np.average(np.array(results), axis=0)
    std_results = np.std(np.array(results), axis=0)
    maximumEpisodes = average_results.shape[0]
    max_avg = np.max(average_results)

    plt.errorbar(np.array([i for i in range(maximumEpisodes)]), average_results, std_results, marker='.', ecolor='aqua')
    plt.grid(True)
    plt.axhline(max_avg)
    plt.text(0, max_avg, "max: "+str(round(max_avg,2)),fontsize=15, backgroundcolor='w')
    
    plt_name = "fchc_gridworld"

    now = datetime.now()
    param_string = "_numTrials_"+str(numTrials)+"_numIter_" + str(numIterations)
    dt_string = now.strftime("_%H_%M")
    
    plt_name+= param_string
    plt_name+= dt_string
    
    print("plt_name=", plt_name)
    plt_path = "images/"+plt_name + ".png"
    
#    plot_min = -60
#    plot_max = 10
#    avg_max = average_results.max()
#    plt.ylim(average_results.min(), average_results.max())
#    plt.ylim(plot_min, plot_max)

    plt_name+= param_string
    plt_name+= dt_string
    print("plt_name=", plt_name)
    plt_path = "images/"+plt_name + ".png"

#    plot_min = -100
#    plot_max = 10
#    plt.ylim(average_results.min(), average_results.max())
#    plt.ylim(plot_min, plot_max)
    
    plt.savefig(plt_path,dpi=200)
    plt.show()
    
    np.save("data/"+"results_"+plt_name, results)
    np.save("data/"+"average_results_"+plt_name, average_results)
    np.save("data/"+"std_results_"+plt_name, std_results)
    

def problem3():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this assignment) on the More-Watery 687-Gridworld domain. Report the same 
    quantities.
    """
       
    def initPopFn(pop_size):
#        theta_arr = np.zeros((pop_size,100))
        theta_zeros = np.zeros(100)
        theta_arr = np.random.multivariate_normal(theta_zeros ,np.identity(len(theta_zeros)),size = pop_size)
#        print(theta_arr.shape)
#        return child
        return theta_arr

    print("ga-gridworld-tabular_softmax")

    popSize = 10
    evaluate = Evaluate()
    numElite = 4
    numEpisodes = 10
#    numEpisodes = 50
    
    ga = GA(popSize , evaluate, initPopFn, numElite, numEpisodes)

#    numTrials = 50
    numTrials = 20
#    numIterations = 100
    numIterations = 25
    
    total_episodes = numIterations*numEpisodes*popSize # 20*50*10

#    total_episodes = numIterations*num_episodes # 100*50
    
    results = np.zeros((numTrials,total_episodes))
#    results = []
#    iter_results = []
    for trial in range(numTrials):
        ga.reset()
        for i in range(numIterations):
            #DEBUG
            if (i%5 == 0):
                print("ga: ","trial: ",trial,"/",numTrials," iteration: ",i,"/",numIterations)
            ga.train()
            
            batch_start = (i*numEpisodes)*popSize
            batch_end = ((i+1)*numEpisodes)*popSize
            results[trial,batch_start:batch_end] =\
                    np.array(evaluate.batchReturn)
#            np.evaluate.batchReturn
            
    average_results = np.mean(np.array(results), axis=0)
    std_results = np.std(np.array(results), axis=0)
    maximumEpisodes = average_results.shape[0]
    max_avg = np.max(average_results)
    
    plt.errorbar(np.array(range(maximumEpisodes)), average_results, std_results, marker='.', ecolor='aqua')
    plt.grid(True)
    plt.axhline(max_avg)
    plt.text(0, max_avg, "max: "+str(round(max_avg,2)),fontsize=15, backgroundcolor='w')
    
    plt_name = "ga_gridworld"

    now = datetime.now()
    param_string = "_numTrials_"+str(numTrials)+"_numIter_" + str(numIterations)
    dt_string = now.strftime("_%H_%M")
    
    plt_name+= param_string
    plt_name+= dt_string
    print("plt_name=", plt_name)
    plt_path = "images/"+plt_name + ".png"

#    plot_min = -100
#    plot_max = 10
#    plt.ylim(average_results.min(), average_results.max())
#    plt.ylim(plot_min, plot_max)
    
    plt.savefig(plt_path,dpi=200)
    plt.show()
    
    np.save("data/"+"results_"+plt_name, results)
    np.save("data/"+"average_results_"+plt_name, average_results)
    np.save("data/"+"std_results_"+plt_name, std_results)
    

def problem4():
    """
    Repeat the previous question, but using the cross-entropy method on the 
    cart-pole domain. Notice that the state is not discrete, and so you cannot 
    directly apply a tabular softmax policy. It is up to you to create a 
    representation for the policy for this problem. Consider using the softmax 
    action selection using linear function approximation as described in the notes. 
    Report the same quantities, as well as how you parameterized the policy. 
    
    """

    #TODO
    
    print("cem-cartpole-softmax_theta_phi")

    state = np.array([0,0,0,0])
    
    env = Cartpole()
    env.nextState(state,0)
    
    fourier_param = 4
    
    theta = np.zeros(2*fourier_param**4) 
    sigma = 1
    popSize = 10
    numElite = 3
    numEpisodes = 5
#    numEpisodes = 20
    evaluate = EvaluateCartpole()
    epsilon = 0.005
    
    cem = CEM(theta,sigma, popSize, numElite, numEpisodes,evaluate,epsilon)

#    numTrials = 50
    numTrials = 10
#    numIterations = 250
    numIterations = 100
    
#    total_episodes = 20,000
    total_episodes = numIterations*numEpisodes*popSize # 20*50*10
    
    results = np.zeros((numTrials,total_episodes))
    
    for trial in range(numTrials):
        cem.reset()
        for i in range(numIterations):
            #DEBUG
            if (i%5 == 0):
                print("cart cem: ","trial: ",trial,"/",numTrials," iteration: ",i,"/",numIterations)
            cem.train()
            
            batch_start = (i*numEpisodes)*popSize
            batch_end = ((i+1)*numEpisodes)*popSize

            results[trial,batch_start:batch_end] = np.array(evaluate.batchReturn)
        
    average_results = np.average(np.array(results), axis=0)
    std_results = np.std(np.array(results), axis=0)
    maximumEpisodes = average_results.shape[0]
    max_avg = np.max(average_results)
  
    plt.errorbar(np.array([i for i in range(maximumEpisodes)]), average_results, std_results, marker='.', ecolor='aqua')
    plt.grid(True)
    plt.axhline(max_avg)
    plt.text(0, max_avg, "max: "+str(round(max_avg,2)),fontsize=15, backgroundcolor='w')

    plt_name = "cem_cartpole"

    now = datetime.now()
    param_string = "_numTrials_"+str(numTrials)+"_numIter_" \
        + str(numIterations) + "_popSize_" +str(popSize)
    dt_string = now.strftime("_t_%H_%M")
    
    plt_name+= param_string
    plt_name+= dt_string
    print("plt_name=", plt_name)
    plt_path = "images/"+plt_name + ".png"

#    plot_min = -100
#    plot_max = 10
#    plt.ylim(average_results.min(), average_results.max())
#    plt.ylim(plot_min, plot_max)
    
    plt.savefig(plt_path,dpi=200)
    plt.show()
    
    np.save("data/"+"results_"+plt_name, results)
    np.save("data/"+"average_results_"+plt_name, average_results)
    np.save("data/"+"std_results_"+plt_name, std_results)

#    pass

def problem5():
    """
    Repeat the previous question, but using first-choice hill-climbing (as 
    described in class) on the cart-pole domain. Report the same quantities 
    and how the policy was parameterized. 
    
    """
    
    print("fchc-cartpole-tabular_softmax")

    fourier_param = 4
    
    theta = np.zeros(2*fourier_param**4) 
    sigma = 1
    num_episodes = 5
    evaluate = EvaluateCartpole()

    fchc = FCHC(theta, sigma, evaluate, num_episodes) # init fchc (calculates J_hat)    
        
    numTrials = 50
#    numTrials = 50
    numIterations = 1000
#    numIterations = 2500
    
#    total_episodes = 20,000
    total_episodes = numIterations*num_episodes # 100*50
    
    results = np.zeros((numTrials,total_episodes))
    
    for trial in range(numTrials):
        fchc.reset()
        for i in range(numIterations):
            #DEBUG
            if (i%5 == 0):
                print("cart fchc: ","trial: ",trial,"/",numTrials," iteration: ",i,"/",numIterations)
            fchc.train()
            
            batch_start = (i*num_episodes)
            batch_end = ((i+1)*num_episodes)
            
            results[trial,batch_start:batch_end] = np.array(evaluate.batchReturn)
        
    average_results = np.average(np.array(results), axis=0)
    std_results = np.std(np.array(results), axis=0)
    maximumEpisodes = average_results.shape[0]
    max_avg = np.max(average_results)
    
    plt.errorbar(np.array([i for i in range(maximumEpisodes)]), average_results, std_results, marker='.', ecolor='aqua')
    plt.grid(True)
    plt.axhline(max_avg)
    plt.text(0, max_avg, "max: "+str(round(max_avg,2)),fontsize=15, backgroundcolor='w')

    plt_name = "fchc_cartpole"

    now = datetime.now()
    param_string = "_numTrials_"+str(numTrials)+"_numIter_" + str(numIterations)
    dt_string = now.strftime("_%H_%M")

    plt_name+= param_string
    plt_name+= dt_string
    print("plt_name=", plt_name)
    plt_path = "images/"+plt_name + ".png"
    
    plt.savefig(plt_path,dpi=200)
    plt.show()
    
    np.save("data/"+"results_"+plt_name, results)
    np.save("data/"+"average_results_"+plt_name, average_results)
    np.save("data/"+"std_results_"+plt_name, std_results)

    #TODO
    pass

def problem6():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this homework) on the cart-pole domain. Report the same quantities and how
    the policy was parameterized. 
    """
    
    print("ga-cartpole-softmax_theta_phi")
#    fourier_param = 2
    def initPopFn(pop_size):
        theta_arr = np.zeros((pop_size,2*fourier_param**4)) 
        return theta_arr

    state = np.array([0,0,0,0])
    
    env = Cartpole()
    env.nextState(state,0)
    
    fourier_param = 4
    
#    theta = np.zeros(2*fourier_param**4) 
#    sigma = 1
    popSize = 10
    numElite = 3
    numEpisodes = 5
    evaluate = EvaluateCartpole()
#    epsilon = 0.005
    
    ga = GA(popSize, evaluate, initPopFn, numElite, numEpisodes)

#    numTrials = 50
    numTrials = 10
    numIterations = 100
#    numIterations = 250
#    numIterations = 20
    total_episodes = numIterations*numEpisodes*popSize # 20*50*10
    
    results = np.zeros((numTrials,total_episodes))
    
    for trial in range(numTrials):
        ga.reset()
        for i in range(numIterations):
            #DEBUG
            if (i%5 == 0):
                print("cart ga: ","trial: ",trial,"/",numTrials," iteration: ",i,"/",numIterations)
            ga.train()
            
            batch_start = (i*numEpisodes)*popSize
            batch_end = ((i+1)*numEpisodes)*popSize

            results[trial,batch_start:batch_end] = np.array(evaluate.batchReturn)
        
    average_results = np.average(np.array(results), axis=0)
    std_results = np.std(np.array(results), axis=0)
    maximumEpisodes = average_results.shape[0]
    max_avg = np.max(average_results)
  
    plt.errorbar(np.array([i for i in range(maximumEpisodes)]), average_results, std_results, marker='.', ecolor='aqua')
    plt.grid(True)
    plt.axhline(max_avg)
    plt.text(0, max_avg, "max: "+str(round(max_avg,2)),fontsize=15, backgroundcolor='w')

    plt_name = "ga_cartpole"

    now = datetime.now()
    param_string = "_numTrials_"+str(numTrials)+"_numIter_" \
        + str(numIterations) + "_popSize_" +str(popSize)
    dt_string = now.strftime("_t_%H_%M")
    
    plt_name+= param_string
    plt_name+= dt_string
    print("plt_name=", plt_name)
    plt_path = "images/"+plt_name + ".png"

#    plot_min = -100
#    plot_max = 10
#    plt.ylim(average_results.min(), average_results.max())
#    plt.ylim(plot_min, plot_max)
    
    plt.savefig(plt_path,dpi=200)
    plt.show()
    
    np.save("data/"+"results_"+plt_name, results)
    np.save("data/"+"average_results_"+plt_name, average_results)
    np.save("data/"+"std_results_"+plt_name, std_results)

    
    #TODO
    pass

def main():
    """
    print("hello, world")
#    problem1() # cem-gridworld-tabular_softmax
#    problem2() # fchc-gridworld-tabular_softmax
#    problem3() # ga-gridworld-tabular_softmax

#    problem4() # cem-cartpole-softmax_theta_phi
#    problem5() # fchc-cartpole-softmax_theta_phi
    problem6() # ga-cartpole-softmax_theta_phi
    
#    env = Cartpole()
#    evaluate_cartpole = EvaluateCartpole()
    """


#    """
##### cem_gridworld ##########################################################    
    average_cem_gridworld = np.load('np_array/cem_gridworld/average_results_cem_gridworld_numTrials_50_numIter_250_popSize_10_t_04_27.npy')
    std_cem_gridworld = np.load('np_array/cem_gridworld/std_results_cem_gridworld_numTrials_50_numIter_250_popSize_10_t_04_27.npy')
##    max_avg_cem_gridworld = np.max(average_cem_gridworld)
##    max_episodes_cem_gridworld= len(average_cem_gridworld)   

#### fchc_gridworld ##########################################################    
    average_fchc_gridworld= np.load('np_array/a/average_results_fchc_gridworld_numTrials_50_numIter_2500_04_40_numTrials_50_numIter_2500_04_40.npy')
    std_fchc_gridworld= np.load('np_array/a/std_results_fchc_gridworld_numTrials_50_numIter_2500_04_40_numTrials_50_numIter_2500_04_40.npy')

##### ga_gridworld ##########################################################    
    average_ga_gridworld = np.load('np_array/ga_gridworld/GA_GRIDWORLD_MEAN20191007-212621.npy')
    std_ga_gridworld = np.load('np_array/ga_gridworld/GA_GRIDWORLD_STD20191007-212621.npy')
#    max_avg_ga_gridworld = np.max(average_ga_gridworld)
#    max_episodes_ga_gridworld = len(average_ga_gridworld)   
#
#### cem_cartpole ##########################################################    
#    average_cem_cartpole = np.load('np_array/cem_cartpole/average_results_cem_cartpole_numTrials_10_numIter_100_popSize_10_t_21_33.npy')
#    std_cem_cartpole = np.load('np_array/cem_cartpole/std_results_cem_cartpole_numTrials_10_numIter_100_popSize_10_t_21_33.npy')
    results_cem_cartpole = np.load('np_array/cem_cartpole/results_cem_cartpole_numTrials_10_numIter_100_popSize_10_t_21_33.npy')
    average_cem_cartpole = np.mean(results_cem_cartpole[:5],axis = 0)
    std_cem_cartpole = np.std(results_cem_cartpole[:5],axis = 0)
#    np.save("average_cem_cartpole",average_cem_cartpole)
#    np.save("std_cem_cartpole",std_cem_cartpole)
##### fchc_cartpole ##########################################################
    results_fchc_cartpole = np.load('np_array/fchc_cartpole/results_fchc_cartpole_numTrials_50_numIter_1000_00_24.npy')
    average_fchc_cartpole  = np.mean(results_fchc_cartpole[:20],axis = 0)
#    np.load('np_array/fchc_cartpole/average_results_fchc_cartpole_numTrials_50_numIter_1000_00_24.npy')
    std_fchc_cartpole = np.std(results_fchc_cartpole[:20],axis = 0)
    np.save("average_fchc_cartpole",average_fchc_cartpole)
    np.save("std_fchc_cartpole",std_fchc_cartpole)

#    results_ga_cartpole = np.load('GA_CARTPOLE_MEAN20191008-205940.npy')
#    print(results_ga_cartpole.shape)
    average_ga_cartpole  = np.load('GA_CARTPOLE_MEAN20191008-205940.npy')
#    np.mean(results_cem_cartpole[:5],axis = 0)
#    np.load('np_array/fchc_cartpole/average_results_fchc_cartpole_numTrials_50_numIter_1000_00_24.npy')
    std_ga_cartpole = np.load('GA_CARTPOLE_STD20191008-205940.npy')
#    np.std(results_cem_cartpole[:5],axis = 0)
#    np.save("average_fchc_cartpole",average_fchc_cartpole)
#    np.save("std_fchc_cartpole",std_fchc_cartpole)

#    np.load('np_array/fchc_cartpole/std_results_fchc_cartpole_numTrials_50_numIter_1000_00_24.npy')
#    results_fchc_cartpole = np.load('results_cem_cartpole_numTrials_10_numIter_100_popSize_10_t_21_33')
#    average_fchc_cartpole = np.mean(results_fchc_cartpole[:25],axis = 0)
#    std_fchc_cartpole = np.std(results_fchc_cartpole[:25],axis = 0)
###### cem_cartpole ##########################################################
#    average_cem_cartpole  = np.load('np_array/fchc_cartpole/average_results_fchc_cartpole_numTrials_50_numIter_1000_00_24.npy')
#    std_cem_cartpole = np.load('np_array/fchc_cartpole/std_results_fchc_cartpole_numTrials_50_numIter_1000_00_24.npy')

#    #GA gridworld plot
#    average = average_ga_gridworld
#    std = std_ga_gridworld
#    max_average = max_avg_ga_gridworld
#    max_episodes = max_episodes_ga_gridworld
#
#    #CEM cartpole plot
#    average = average_cem_cartpole 
#    std = std_cem_cartpole 
#    max_average = max_avg_cem_cartpole
#    max_episodes = max_episodes_cem_cartpole
#    
#    # FCHC cartpole plot
#    average = average_fchc_cartpole 
#    std = std_fchc_cartpole 
#    max_average = max_avg_fchc_cartpole
#    max_episodes = max_episodes_fchc_cartpole
    
#    size = len(average_fchc_gridworld)
#    max_episodes = min(10000,size)
#    average = average_fchc_gridworld[:max_episodes]
#    std = std_fchc_gridworld[:max_episodes]
#    max_average = np.max(average)    
#    
    size = len(average_ga_gridworld)
    max_episodes = min(8000,size)
    average = average_ga_gridworld[:max_episodes]
    std = std_ga_gridworld[:max_episodes]
    max_average = np.max(average)
        
    #plot
    plt.errorbar(np.array([i for i in range(max_episodes)]), average, std, marker='.', ecolor='aqua')
    plt.grid(True)
    plt.axhline(max_average)
    plt.text(0, max_average, str(round(max_average,2)),fontsize=10, backgroundcolor='w')
    plt.savefig("images/average_ga_gridworld"+".png",dpi=200)
    plt.show()

#    """

if __name__ == "__main__":
    main()
