# Inteligencia Artifical aplicada a Negocios y Empresas

# Maximizacion de beneficios de una empresa de venta online con Muestreo de Thompson


## Import libraries



import numpy as np
import matplotlib.pyplot as plt
import random


## Config Parameters

C = 100000 #Number of rounds = Number of clients
s = 10 #Number of marketing strategies


## Building the simulation 

conversion_rates = [0.01, 0.08, 0.1, 0.04, 0.21, 0.12, 0.17, 0.09, 0.12, 0.25]
#conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
X = np.array(np.zeros([C, s]))
for i in range(C): #Rounds=clients
    for j in range(s): #Strategy
        if np.random.rand() <= conversion_rates[j]: #if our random number.....then...
            X[i,j] = 1


## Random assignment and Thompson estimator

strategies_selected_rs = [] #random select
strategies_selected_ts = [] #thompson select
total_reward_rs = 0
total_reward_ts = 0
number_of_rewards_1 = [0] * s #creating a s-zeros list
number_of_rewards_0 = [0] * s
for n in range(0, C): #Starting rounds=clients
    # Random Selection
    strategy_rs = random.randrange(s) #select a random strategy
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs] #WE get the reward which has been created before
    total_reward_rs += reward_rs
    # Thompson Estimator
    strategy_ts = 0
    max_random = 0
    for i in range(0, s):
        #Step 1
        random_beta = random.betavariate(number_of_rewards_1[i]+1, 
                                         number_of_rewards_0[i]+1)
        #Step 2
        if random_beta > max_random: 
            max_random = random_beta
            strategy_ts = i
    strategies_selected_ts.append(strategy_ts)
    #Step 3
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        number_of_rewards_1[strategy_ts] += 1
    else:
        number_of_rewards_0[strategy_ts] += 1
    total_reward_ts += reward_ts


## Relative and absolute return

absolute_return = (total_reward_ts - total_reward_rs)*100 #We assume 100 $/sale.
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100
print("Absolute Permormance: {:.0f} $".format(absolute_return))
print("Relative Permormance: {:.0f} %".format(relative_return))


# Plotting histogram

plt.hist(strategies_selected_ts)
plt.title("Histogram of selections")
plt.xlabel("Strategy")
plt.ylabel("Number of times has been selected the strategy")
plt.show()


    

