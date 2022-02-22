
import torch
from experiment_weighted_or import MaxLength
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arm_run():
    def __init__(self, arm):
        self.arm = arm

        return

    def run(self):
        reward, collected  = MaxLength.reward(self, max_trajectory=50, weight=arm/10)
        return  reward, collected


iter = 2000
mean_reward = 0

#Create a DataFrame to collect the rewards for each arm
s = np.zeros(11)
error = np.zeros(11)

for arm in range(11):
    x = np.zeros(iter)
    for steps in range(iter):
        a = Arm_run(arm)
        reward, collected = a.run()
        x[steps] = reward
        # Update total
        mean_reward = mean_reward + (reward - mean_reward) / (steps + 1)

    #Add results to the DataFrames
    s[arm] = mean_reward
    error[arm] = np.std(x)


plt.title("Average rewards per Arm after 2000 steps")
plt.xlabel('Arms', fontweight='bold', fontsize=15)
plt.ylabel('Average Rewards', fontweight='bold', fontsize=15)
plt.bar(np.arange(11), s, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
#plt.bar(x=np.arange(11), height=s,)
plt.show()

###############################################################################################################################################
#HIDING SOME CODE UCB

k = 11  # number of arms
iters = 2

#Parameter settings
c = [0, 1, np.sqrt(2), 2, 3]
seeds = [0, 43, 51, 79, 101]

#Create list with 5 variable dataframes for each parameter gamma
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

    for x, df, df_a in zip(c, data_frames, df_arms):
        # Initialize bandits
        ucb = Ucb(k, x, iters)
        ucb_rewards = np.zeros(iters)

        # Run experiments
        ucb.run()

        # Obtain stats
        ucb_rewards = ucb.reward
        ucb_collected = ucb.collected_items
        arms = pd.DataFrame([ucb.k_n - 1])

        #Add results to the DataFrames
        exec('{}[n_experiment] = ucb_rewards'.format(df))
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
plt.plot(x, mean_1, 'b-', label='0')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, mean_2, 'r-', label='1')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.plot(x, mean_3, 'k-', label='√2')
plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
plt.plot(x, mean_4, 'g-', label='2')
plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
plt.plot(x, mean_5, 'y-', label='3')
plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
plt.legend(title=r'$\c$', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("11-armed bandits: UCB")
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
plt.bar(br1, mean_arm_1, color='b', width=barWidth, edgecolor='grey', label='0')
plt.bar(br2, mean_arm_2, color='r', width=barWidth, edgecolor='grey', label='1')
plt.bar(br3, mean_arm_3, color='k', width=barWidth, edgecolor='grey', label='√2')
plt.bar(br4, mean_arm_4, color='g', width=barWidth, edgecolor='grey', label='2')
plt.bar(br5, mean_arm_5, color='y', width=barWidth, edgecolor='grey', label='3')


# Adding Xticks
plt.title("Arm sampling distribution: UCB")
plt.xlabel('Arms', fontweight='bold', fontsize=15)
plt.ylabel('Average Pulls', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(mean_arm_1))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.legend(title=r'$\c$', loc='best')
plt.show()

#####################################################################################################################################################################################
#Thompson Sampling

k = 11  # number of arms
iters = 2

#Parameter settings
means = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
std_deviations = [1, 10, 50, 100, 1000]
seeds = [0, 43, 51, 79, 101]

#Create list with 5 variable dataframes for each parameter gamma
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

    for x, df, df_a in zip(zip(means, std_deviations), data_frames, df_arms):
        # Initialize bandits
        ts = ThompsonSampling(k, iters, x[0], x[1])
        ts_rewards = np.zeros(iters)

        # Run experiments
        ts.run()

        # Obtain stats
        ts_rewards = ts.reward
        ts_collected = ts.collected_items
        arms = pd.DataFrame([ts.counts])

        #Add results to the DataFrames
        exec('{}[n_experiment] = ts_rewards'.format(df))
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
plt.plot(x, mean_1, 'b-', label='0 & 1')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, mean_2, 'r-', label='0.2 & 10')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.plot(x, mean_3, 'k-', label='0.4 & 50')
plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
plt.plot(x, mean_4, 'g-', label='0.6 & 100')
plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
plt.plot(x, mean_5, 'y-', label='0.8 & 1000')
plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
plt.legend(title=r'$\mu$'+' & '+'$\sigma$', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("11-armed bandits:  Gaussian Thompson Sampling")
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
plt.bar(br1, mean_arm_1, color='b', width=barWidth, edgecolor='grey', label='0 & 1')
plt.bar(br2, mean_arm_2, color='r', width=barWidth, edgecolor='grey', label='0.2 & 10')
plt.bar(br3, mean_arm_3, color='k', width=barWidth, edgecolor='grey', label='0.4 & 50')
plt.bar(br4, mean_arm_4, color='g', width=barWidth, edgecolor='grey', label='0.6 & 100')
plt.bar(br5, mean_arm_3, color='y', width=barWidth, edgecolor='grey', label='0.8 & 1000')

#Adding Xticks
plt.title("Arm sampling distribution: Gaussian Thompson Sampling")
plt.xlabel('Arms', fontweight='bold', fontsize=15)
plt.ylabel('Average Pulls', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(mean_arm_1))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.legend(title=r'$\mu$'+' & '+'$\sigma$', loc='best')
plt.show()


#############################################################################################################################################################
###Softmax


k = 11  # number of arms
iters = 2

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
plt.title("Arm sampling distribution: Softmax")
plt.xlabel('Arms', fontweight='bold', fontsize=15)
plt.ylabel('Average Pulls', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(mean_arm_1))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.legend(title=r'$\tau$', loc='best')
plt.show()


#############################################################################################################################################
#EXP 3
k = 11  # number of arms
iters = 2

#Parameter settings
gamma = [0.1, 0.2, 0.3, 0.4, 0.5]
seeds = [0, 43, 51, 79, 101]

#Create list with 5 variable dataframes for each parameter gamma
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

    for x, df, df_a in zip(gamma, data_frames, df_arms):
        # Initialize bandits
        exp3 = Exp3(x, iters, k)
        exp3_rewards = np.zeros(iters)

        # Run experiments
        exp3.run()

        # Obtain stats
        exp3_rewards = exp3.reward
        exp3_collected = exp3.collected_items
        arms = pd.DataFrame([exp3.counts])

        #Add results to the DataFrames
        exec('{}[n_experiment] = exp3_rewards'.format(df))
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
plt.legend(title=r'$\gamma$', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("11-armed bandits: Exp3")
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

#Adding Xticks
plt.title("Arm sampling distribution: Exp3")
plt.xlabel('Arms', fontweight='bold', fontsize=15)
plt.ylabel('Average Pulls', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(mean_arm_1))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.legend(title=r'$\gamma$', loc='best')
plt.show()



