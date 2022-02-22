#https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-softmax-algorithm-e1fa4cb0c422

import math
import random  # Arm selection based on Softmax probability
import torch
import numpy as np
from experiment_weighted_or import MaxLength
import matplotlib.pyplot as plt
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Softmax:
    def __init__(self, tau, iters, n_arms):
        self.tau = tau
        # Number of arms
        self.n_arms = n_arms
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        self.counts = [0 for col in range(n_arms)]  # Count represent counts of pulls for each arm. For multiple arms, this will be a list of counts.
        self.values = [0.0 for col in range(n_arms)]  # Value represent average reward for specific arm. For multiple arms, this will be a list of values.
        self.collected_items = [] # Items collected
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        return

    def categorical_draw(self, probs):
        z = random.random()
        cum_prob = 0.0

        for i in range(len(probs)):
            prob = probs[i]
            cum_prob += prob

            if cum_prob > z:
                return i
        return len(probs) - 1  # Softmax algorithm

    def select_arm(self):
        # Calculate Softmax probabilities based on each round
        z = sum([math.exp(v / self.tau) for v in self.values])
        probs = [math.exp(v / self.tau) / z for v in self.values]

        # Use categorical_draw to pick arm
        return self.categorical_draw(probs)

    def pull(self):
        #Select highest preference action
        chosen_arm = self.select_arm()

        #Get the reward and item collected
        reward, collected = MaxLength.reward(self, max_trajectory=50, weight=chosen_arm/10)

        # Update counts
        self.n += 1

        #Update the chosen_arm and reward
        self.update(chosen_arm, reward, collected)

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward, collected):
        # update counts pulled for chosen arm
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        # Update total
        self.mean_reward = self.mean_reward + (
                reward - self.mean_reward) / self.n

        # Update average/mean value/reward for chosen arm
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

        # Update the item collected after the pull
        self.collected_items.append(collected)
        return

k = 11  # number of arms
iters = 1000

#Parameter settings
tau = [0.1, 0.2, 0.3, 0.4, 0.5]
seeds = [0, 43, 51, 79, 101]

#Create list with 5 variable dataframes for each temperature
data_frames = ['df1', 'df2', 'df3', 'df4', 'df5'] #To store rewards
df_arms = ['df_arms_1', 'df_arms_2', 'df_arms_3', 'df_arms_4', 'df_arms_5']

# Create empty dataframes to store average rewards for each seed per temperature
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()

# Create empty dataframes to store the number of arms collected per temperature
df_arms_1 = pd.DataFrame()
df_arms_2 = pd.DataFrame()
df_arms_3 = pd.DataFrame()
df_arms_4 = pd.DataFrame()
df_arms_5 = pd.DataFrame()



for n_experiment in seeds:
    np.random.seed(n_experiment)
    random.seed(n_experiment)

    for x, df, df_a in zip(tau, data_frames, df_arms):
        # Initialize bandits
        softmax = Softmax(x, iters, k)
        softmax_rewards = np.zeros(iters)

        # Run experiments
        softmax.run()

        # Obtain stats
        softmax_rewards = softmax.reward
        softmax_collected = softmax.collected_items
        arms = pd.DataFrame([softmax.counts])

        #Add results to the DataFrames
        exec('{}[n_experiment] = softmax_rewards'.format(df))
        exec('{} = pd.DataFrame({}).append(arms, ignore_index=True)'.format(df_a, df_a))

#Calculate the average mean and standard deviation curves
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
plt.plot(x, mean_1, 'b-', label='0.1')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, mean_2, 'r-', label='0.2')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.plot(x, mean_3, 'k-', label='0.3')
plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
plt.plot(x, mean_4, 'g-', label='0.4')
plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
plt.plot(x, mean_5, 'y-', label='0.5')
plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
plt.ylim(-7.5, -1.0)
plt.legend(title=r'$\tau$', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("11-armed bandits: Softmax")
plt.show()

#Calculate the average arms pulled per temperature
mean_arm_1 = df_arms_1.mean()
mean_arm_2 = df_arms_2.mean()
mean_arm_3 = df_arms_3.mean()
mean_arm_4 = df_arms_4.mean()
mean_arm_5 = df_arms_5.mean()

#Make the plot for the mean arms
barWidth = 0.1
# Set position of bar on X axis
br1 = np.arange(len(mean_arm_1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
#mean_arm_1.index
plt.bar(br1, mean_arm_1, color='b', width=barWidth, edgecolor='grey', label='0.1')
plt.bar(br2, mean_arm_2, color='r', width=barWidth, edgecolor='grey', label='0.2')
plt.bar(br3, mean_arm_3, color='k', width=barWidth, edgecolor='grey', label='0.3')
plt.bar(br4, mean_arm_4, color='g', width=barWidth, edgecolor='grey', label='0.4')
plt.bar(br5, mean_arm_3, color='y', width=barWidth, edgecolor='grey', label='0.5')

# Adding Xticks
plt.title("Weight sampling distribution: Softmax")
plt.xlabel('Weight', fontweight='bold', fontsize=15)
plt.ylabel('Average Pulls', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(mean_arm_1))], ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
plt.legend(title=r'$\tau$', loc='best')
plt.show()

results = open("Softmax_Results", "w")
table = np.column_stack((mean_2, std_2))
np.savetxt(results, table, fmt='%.3e', delimiter="  ")
results.close()