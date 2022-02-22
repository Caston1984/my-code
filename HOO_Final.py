from __future__ import division
import numpy as np
import random

# Definitions of bandit algorithms
from lghoo import *

from functions import *
from arms import *

import pandas as pd
import sys
import pickle

#simple progress bar
import pyprind

import torch
from experiment_weighted_or import MaxLength
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import time


#Threading
from threading import Thread


## Right now the simulation works only for underlying functions between 0 and 1
# and with rewards between 0 and 1



def test_algorithm(arm_range, horizon, underfunc, v1, rho, save=False, num_sim=1, height_limit=10, minimum_grow=20, best_arm_policy='new'):
    """
    This function runs the LGHOO algorithm
    :param arm_range: range of the algorithm, with the current underlying functions it is only between 0 and 1
    :param horizon: horizon of the algorithm
    :param underfunc: one of the underlying functions
    :param plot: True if we want to plot the graph only the LAST run. If plot is false it will return the metric for the metrics
    :param save: if we want to save the graph the plotted graph of the LAST run
    :param num_sim: number of times we will run the whole algorithm (for monte carlo)
    :param height_limit:
    :param minimum_grow:
    :param best_arm_policy: type of policy for selecting the best arm
    :return:
    """
    print("Starting the algorithm")
    bar = pyprind.ProgBar(num_sim, stream=sys.stdout)
    #creating the LGHOO object
    algo = []

    # initial vectors representing the variables that will be returned
    regret = np.zeros(num_sim)
    cumulative_rewards = np.zeros(num_sim)
    euclidian_distance = np.zeros(num_sim)
    time_spent = np.zeros(num_sim)
    object_size = np.zeros(num_sim)

    #starting with 1 simulation
    #algo.initialize(len(arms))
    func = []

    algo = []
    #reset the algorithm every round
    algo = LGHOO(arm_range, height_limit=height_limit, v1=v1, rho=rho, minimum_grow=minimum_grow)
    bar.update()

    # Underlying function
    #reevaluating it so we get updates
    func = underfunc().eval
    x_axis, y_axis = generate_xy(func, [algo.arm_range_min, algo.arm_range_max])
    xmax, ymax = getMaxFunc(x_axis, y_axis)
    max_reward = 1.0

    #t0 = time.time()
    mean_reward = 0
    z_rewards = np.zeros(horizon)
    active_arms = []
    for t in range(horizon):
        # each arm in the simulation
        index = t
        arm = algo.select_arm()
        #print('Selected Ruoko', arm)

        #choice of the underlying distribution
        #reward = BernoulliArm(func(arm))
        m = MaxLength(gym.Wrapper, 50)
        reward, collected = m.reward(max_trajectory=50, weight=arm)

        # Update total
        mean_reward = mean_reward + (reward - mean_reward) / (t + 1)
        z_rewards[t] = mean_reward
        active_arms.append(arm)
        #Get the cumulative reward
        #cumulative_rewards[i] = cumulative_rewards[i] + reward

        #update the algorithm
        algo.update(arm, reward)
    #t1 = time.time()
    #time_spent[i] = t1-t0

    best_arm = []
    #Get the best arm - Original selection of the higher node of the tree
    if best_arm_policy == 'original':
        best_arm = algo.get_original_best_arm()
    if best_arm_policy == 'new':
        best_arm = algo.get_best_arm_value()
    #calculate the distance (not perfect specially if there is more than one maximum)
    euclidian_distance = np.absolute(xmax-best_arm)

    #calculate the regret
    #max_exp_value = ymax*max_reward*horizon
    #regret[i] = max_exp_value - cumulative_rewards[i]

    #log the object size
    #object_size[i] = sys.getsizeof(algo)

#if plot==True:
    # Underlying function
  #  x_axis, y_axis = generate_xy(func, [algo.arm_range_min, algo.arm_range_max])
   #     # xmax, ymax = getMaxFunc(x_axis, y_axis)
    #    filename = underfunc.__name__+"-"+str(horizon)+".png"
     #   algo.plot_graph_with_function(x_axis,y_axis,rescale_y=3, save=save, filename=filename)
      #  return
    #else:
    return z_rewards, active_arms #, active_arms #, cumulative_rewards, euclidian_distance, regret, time_spent, object_size]

# Total mean reward
trials = 10

#reward = np.zeros(trials)

# Parameter settings
v1 = [0.2, 0.4, 0.5, 0.7, 1.0]
rho = [0.8, 0.6, 0.5, 0.4, 0.3]
seeds = [0, 43, 51, 79, 101]

# Create list with 5 variable dataframes for each temperature
data_frames = ['df1', 'df2', 'df3', 'df4', 'df5']  # To store rewards
df_arms = ['df_arms_1', 'df_arms_2', 'df_arms_3', 'df_arms_4', 'df_arms_5'] # To store arms pulled

# Create empty dataframes to store average rewards for each seed per temperature
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()

# Create empty dataframes to store the number of arms collected per temperature
a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20, a_21, a_22, a_23, a_24, a_25 = [[] for _ in range(25)]

a=0
for n_experiment in seeds:
    np.random.seed(n_experiment)
    random.seed(n_experiment)

    for v_1, rho_, df, df_a in zip(v1, rho, data_frames, df_arms):
        # Initialize bandits


        avg_rewards, active_arms = test_algorithm(arm_range = [0, 1], horizon = trials, underfunc = normal, v1=v_1, rho=rho_ , save=False, num_sim=1, height_limit=10,
                       minimum_grow=20, best_arm_policy='new')
        print('Active arms', active_arms)

        a += 1


        # Add results to the DataFrames
        exec('{}[n_experiment] = avg_rewards'.format(df))
        exec('a_{} = a_{}.append(active_arms)'.format(a, a))
        #exec('{} = pd.DataFrame({}).append(tried_arms), ignore_index=True)'.format(df_a, df_a))




print(a_1)
# Calculate the average mean and standard deviation curves
mean_1 = df1.mean(axis=1)
std_1 = df1.std(axis=1)
mean_2 = df2.mean(axis=1)
std_2 = df2.std(axis=1)
mean_3 = df3.mean(axis=1)
std_3 = df3.std(axis=1)
mean_4 = df4.mean(axis=1)
std_4 = df4.std(axis=1)
mean_5 = df5.mean(axis=1)
std_5 = df5.std(axis=1)

x = np.arange(len(mean_1))
plt.plot(x, mean_1, 'b-', label=r'$v_1 = 0.2 & \rho = 0.8$')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, mean_2, 'r-', label=r'$v_1 = 0.4 & \rho = 0.6$')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.plot(x, mean_3, 'k-', label=r'$v_1 = 0.5 & \rho = 0.5$')
plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
plt.plot(x, mean_4, 'g-', label=r'$v_1 = 0.7 & \rho = 0.4$')
plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
plt.plot(x, mean_5, 'y-', label=r'$v_1 = 0.3 & \rho = 0.9$')
plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
plt.legend(title=r'$v_1 & \rho$', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("HOO Algorithm")
plt.show()


results = open("HOO_Results", "w")
table = np.column_stack((mean_4, std_4))
np.savetxt(results, table, fmt='%.3e', delimiter="  ")
results.close()

mean_a1 = np.mean(a_1, axis=0)
std_a1 = np.std(a_1)
mean_a6 = np.mean(a_6)
std_a6 = np.std(a_6)
mean_a11 = np.mean(a_11)
std_a11 = np.std(a_11)
mean_a16 = np.mean(a_16)
std_a16 = np.std(a_16)
mean_a21 = np.mean(a_21)
std_a21 = np.std(a_21)

x_pos = np.arange(5) #is the array with the count of the number of bars.
CTEs = [mean_a1, mean_a6, mean_a11, mean_a16, mean_a21] # is our array which contains the means or heights of the bars.
error = [std_a1, std_a6, std_a11, std_a16, std_a21] # error sets the heights of the error bars and the standard deviations.

seed = ['0', '43', '51', '79', '101']
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Weight')
ax.set_xticks(x_pos)
ax.set_xticklabels(seed)
ax.set_title('HOO Algorithm: ' r'$v_1 = 0.2 & \rho = 0.8'' & Seed = [0, 43, 51, 79, 101]')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('HOO 1.png')
plt.show()


mean_a2 = np.mean(a_2)
std_a2 = np.std(a_2)
mean_a7 = np.mean(a_7)
std_a7 = np.std(a_7)
mean_a12 = np.mean(a_12)
std_a12 = np.std(a_12)
mean_a17 = np.mean(a_17)
std_a17 = np.std(a_17)
mean_a22 = np.mean(a_22)
std_a22 = np.std(a_22)

x_pos = np.arange(5) #is the array with the count of the number of bars.
CTEs = [mean_a2, mean_a7, mean_a12, mean_a17, mean_a22] # is our array which contains the means or heights of the bars.
error = [std_a2, std_a7, std_a12, std_a17, std_a22] # error sets the heights of the error bars and the standard deviations.
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Weight')
ax.set_xticks(x_pos)
ax.set_xticklabels(seed)
ax.set_title('HOO Algorithm: ' r'$v_1 = 0.4 & \rho = 0.6'' & Seed = [0, 43, 51, 79, 101]')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('HOO 2.png')
plt.show()

mean_a3 = np.mean(a_3)
std_a3 = np.std(a_3)
mean_a8 = np.mean(a_8)
std_a8 = np.std(a_8)
mean_a13 = np.mean(a_13)
std_a13 = np.std(a_13)
mean_a18 = np.mean(a_18)
std_a18 = np.std(a_18)
mean_a23 = np.mean(a_23)
std_a23 = np.std(a_23)


x_pos = np.arange(5)  # is the array with the count of the number of bars.
CTEs = [mean_a3, mean_a8, mean_a13, mean_a18,
            mean_a23]  # is our array which contains the means or heights of the bars.
error = [std_a3, std_a8, std_a13, std_a18,
             std_a23]  # error sets the heights of the error bars and the standard deviations.

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Weight')
ax.set_xticks(x_pos)
ax.set_xticklabels(seed)
ax.set_title('HOO Algorithm: ' r'$v_1 = 0.5 & \rho = 0.5'' & Seed = [0, 43, 51, 79, 101]')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('HOO 3.png')
plt.show()

mean_a4 = np.mean(a_4)
std_a4 = np.std(a_4)
mean_a9 = np.mean(a_9)
std_a9 = np.std(a_9)
mean_a14 = np.mean(a_14)
std_a14 = np.std(a_14)
mean_a19 = np.mean(a_19)
std_a19 = np.std(a_19)
mean_a24 = np.mean(a_24)
std_a24 = np.std(a_24)

x_pos = np.arange(5)  # is the array with the count of the number of bars.
CTEs = [mean_a4, mean_a9, mean_a14, mean_a19,
            mean_a24]  # is our array which contains the means or heights of the bars.
error = [std_a4, std_a9, std_a14, std_a19,
             std_a24]  # error sets the heights of the error bars and the standard deviations.

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Weight')
ax.set_xticks(x_pos)
ax.set_xticklabels(seed)
ax.set_title('HOO Algorithm: ' r'$v_1 = 0.7 & \rho = 0.4'' & Seed = [0, 43, 51, 79, 101]')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('HOO 4.png')
plt.show()

mean_a5 = np.mean(a_5)
std_a5 = np.std(a_5)
mean_a10 = np.mean(a_10)
std_a10 = np.std(a_10)
mean_a15 = np.mean(a_15)
std_a15 = np.std(a_15)
mean_a20 = np.mean(a_20)
std_a20 = np.std(a_20)
mean_a25 = np.mean(a_25)
std_a25 = np.std(a_25)

x_pos = np.arange(5)  # is the array with the count of the number of bars.
CTEs = [mean_a5, mean_a10, mean_a15, mean_a20,
            mean_a25]  # is our array which contains the means or heights of the bars.
error = [std_a5, std_a10, std_a15, std_a20,
             std_a25]  # error sets the heights of the error bars and the standard deviations.

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Weight')
ax.set_xticks(x_pos)
ax.set_xticklabels(seed)
ax.set_title('HOO Algorithm: ' r'$v_1 = 0.7 & \rho = 0.4''= √4 & Seed = [0, 43, 51, 79, 101]')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('alpha √4.png')
plt.show()

m_1 = np.mean([mean_a1, mean_a6, mean_a11, mean_a16, mean_a21])
s_1 = np.mean([std_a1, std_a6, std_a11, std_a16, std_a21])
m_2 = np.mean([mean_a2, mean_a7, mean_a12, mean_a17, mean_a22])
s_2 = np.mean([std_a2, std_a7, std_a12, std_a17, std_a22])
m_3 = np.mean([mean_a3, mean_a8, mean_a13, mean_a18, mean_a23])
s_3 = np.mean([std_a3, std_a8, std_a13, std_a18, std_a23])
m_4 = np.mean([mean_a4, mean_a9, mean_a14, mean_a19, mean_a24])
s_4 = np.mean([std_a4, std_a9, std_a14, std_a19, std_a24])
m_5 = np.mean([mean_a5, mean_a10, mean_a15, mean_a20, mean_a25])
s_5 = np.mean([std_a5, std_a10, std_a15, std_a20, ])

x_pos = np.arange(5)  # is the array with the count of the number of bars.
CTEs = [m_1, m_2, m_3, m_4, m_5]  # is our array which contains the means or heights of the bars.
error = [s_1, s_2, s_3, s_4, s_5]  # error sets the heights of the error bars and the standard deviations.

parameters = [r'$v_1 = 0.2 & \rho = 0.8', r'$v_1 = 0.4 & \rho = 0.6', r'$v_1 = 0.5 & \rho = 0.5', r'$v_1 = 0.7 & \rho = 0.4', r'$v_1 = 0.9 & \rho = 0.3']

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Weight')
ax.set_xticks(x_pos)
ax.set_xticklabels(parameters)
ax.set_title('HOO Algorithm: Average arm selected')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('alpha.png')
plt.show()
