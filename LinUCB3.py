# Do not edit. These are the only imports permitted.
# https://github.com/ChristianLan1/Multi-Armed_Bandit/blob/master/xinjiel2.ipynb
#%matplotlib inline
import numpy as np
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import torch
import pandas as pd
from experiment_weighted_or import MaxLength
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initial three lists which respects
# to arm, reward, and context(feature_list in this case).
arm_list = []
reward_list = []
features_list = []
num_of_events = 0

# Read each line and split by spaces. Record arm, reward and context into the lists
with open("dataset.txt", "r") as f:
    dataset = f.readlines()
for line in dataset:
    num_of_events += 1
    temp_line = line.split()
    arm = int(temp_line[0])
    reward = float(temp_line[1])
    features = temp_line[2:]
    features = list(map(float, features))
    arm_list.append(arm)
    reward_list.append(reward)
    features_list.append(features)

# Convert lists into np_array
arms = np.array(arm_list)
rewards = np.array(reward_list)

# For each event, the context is 10*10 dim
# because there are 10 arms, and each one of them has 10 features
contexts = np.array(features_list).reshape(num_of_events, (10 * 10))
print(arms)
# rewards, removed parameter
def offlineEvaluate(mab, arms, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """
    assert (arms.shape == (num_of_events,)), "1d array and in range[1...map.narms]"
    #assert (rewards.shape == (num_of_events,)), "must be 1d array"
    assert (nrounds > 0 or nrounds is None), "must be positive integer with default None"
    h0 = []  # History list
    R0 = []  # Total Payoff
    rewards = np.zeros(nrounds)

    count = 0
    for event in range(num_of_events):
        # If reach required number of rounds then stop
        # If number of rounds not specified, then read untill last of events.
        if len(h0) == nrounds:
            break
        # Play an arm, but the tround is the number of history observed
        action = mab.play(len(h0) + 1, contexts[event])

        # If the chosen arm is equal to the arm in the log,
        # then record history and payoff, and also update the arm.
        if action == arms[event]:
            count += 1
            h0.append(event)
            c = MaxLength(gym.Wrapper, 50)
            reward, collected = c.reward(max_trajectory=50, weight=action / 10)
            R0.append(reward)
            mab.update(arms[event], reward, collected, contexts[event])

            rewards[count-1] = mab.mean_reward
            print(rewards)
    return R0, rewards

class MAB(ABC):
    """
    Abstract class that represents a multi-armed bandit (MAB)
    """

    @abstractmethod
    def play(self, tround, context):
        """
        Play a round

        Arguments
        =========
        tround : int
            positive integer identifying the round

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to the arms

        Returns
        =======
        arm : int
            the positive integer arm id for this round
        """

    @abstractmethod
    def update(self, arm, reward, context):
        """
        Updates the internal state of the MAB after a play

        Arguments
        =========
        arm : int
            a positive integer arm id in {1, ..., self.narms}

        reward : float
            reward received from arm

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to arms
        """

class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    alpha : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, alpha):
        assert (narms > 0), "narms must be positive integers"
        assert (ndims > 0), "ndims must be positive integers"
        assert ((type(alpha) == float or type(alpha) == np.float64) and alpha > 0.0 and np.isreal(
            alpha)), "alpha must be real positive floating number"
        self.narms = narms
        self.ndims = ndims
        self.alpha = alpha
        self. counts = [0 for col in range(narms)]  # Count represent counts of pulls for each arm. For multiple arms, this will be a list of counts.
        self.values = [0.0 for col in range(narms)]  # Value represent average reward for specific arm. For multiple arms, this will be a list of values.
        self.collected_items = []  # Items collected
        # Total mean reward
        self.mean_reward = 0

        self.A_a = {}  # A is the list of each arm with D.T * D + I
        self.b_a = {}  # b is the reward list

        for arm in range(1, self.narms + 1):
            if arm not in self.A_a:  # If arm is new
                # For each arm, initial identity matrix with feature dimensitonal space
                self.A_a[arm] = np.identity(self.ndims)
            if arm not in self.b_a:
                # For each arm, initial reward matrix with zeros. Dimension is each corresponding context's dimension
                self.b_a[arm] = np.zeros(self.ndims)

    def play(self, tround, context):
        assert (tround > 0), "tround must be positive integers"

        assert (context.shape == (
        self.narms * self.ndims,)), "context must be a numeric array of length self.ndims * self.narms"
        arm_with_Q = {}  # At tround, initial arm with empty posterior distribution

        context = context.reshape(self.narms, self.ndims)

        for arm in range(1, self.narms + 1):
            # For each arm, calculate posterior distribution based on theta and std
            Theta_a = np.dot(np.linalg.inv(self.A_a[arm]), self.b_a[arm])
            std = np.sqrt(
                np.linalg.multi_dot([np.transpose(context[arm - 1]), np.linalg.inv(self.A_a[arm]), context[arm - 1]]))
            p_ta = np.dot(Theta_a.T, context[arm - 1]) + self.alpha * std

            if not np.isnan(p_ta):  # make sure the result of calculation is valid number
                arm_with_Q[arm] = p_ta

        # Getting the highest value from posterior distribution, then find the corresponding key and append them
        highest = max(arm_with_Q.values())

        highest_Qs = [key for key, value in arm_with_Q.items() if value == highest]
        if len(highest_Qs) > 1:
            action = np.random.choice(highest_Qs)  # Tie Breaker
        else:
            action = highest_Qs[0]

        return action

    def update(self, arm, reward, collected, context):
        assert (arm > 0 and arm <= self.narms), "arm must be positive integers and no larger than self.narms"
        assert (type(reward) == float or type(reward) == np.float64), "reward must be floating point"

        assert (context.shape == (
        self.narms * self.ndims,)), "context must be a numeric array of length self.ndims * self.narms"


        context = context.reshape(self.narms, self.ndims)

        if arm <= self.narms:
            # Reshap the vector to matrix, or the calculation will be incorrect
            # because the transpose will not take effects
            context_Matrix = context[arm - 1].reshape(-1, 1)
            context_times_contextT = np.dot(context_Matrix, context_Matrix.T)

            self.A_a[arm] = np.add(self.A_a[arm], context_times_contextT)

            self.b_a[arm] = np.add(self.b_a[arm], np.dot(reward, context[arm - 1]))

            # update counts pulled for chosen arm
            self.counts[arm-1] = self.counts[arm-1] + 1
            n = self.counts[arm-1]

            # Update total
            self.mean_reward = self.mean_reward + (reward - self.mean_reward) / n

            # Update average/mean value/reward for chosen arm
            value = self.values[arm-1]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            self.values[arm-1] = new_value

            # Update the item collected after the pull
            self.collected_items.append(collected)


k = 10  # number of arms
iters = 1000

#Parameter settings
alpha = [0.5, np.sqrt(1.0), np.sqrt(2.0), np.sqrt(3.0), np.sqrt(4.0)]
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

    for x, df, df_a in zip(alpha, data_frames, df_arms):
        # Initialize bandits
        mab = LinUCB(10, 10, x)

        # Run experiments
        results_LinUCB, rewards = offlineEvaluate(mab, arms, contexts, iters)

        # Obtain stats
        LinUCB_rewards = rewards
        #softmax_collected = softmax.collected_items
        #arms = pd.DataFrame([softmax.counts])

        #Add results to the DataFrames
        exec('{}[n_experiment] = LinUCB_rewards'.format(df))
        #exec('{} = pd.DataFrame({}).append(arms, ignore_index=True)'.format(df_a, df_a))

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
plt.plot(x, mean_1, 'b-', label='0.5')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, mean_2, 'r-', label=r'$\sqrt{1.0}$ ')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.plot(x, mean_3, 'k-', label=r'$\sqrt{2.0}$ ')
plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
plt.plot(x, mean_4, 'g-', label=r'$\sqrt{3.0}$ ')
plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
plt.plot(x, mean_5, 'y-', label=r'$\sqrt{4.0}$ ')
plt.fill_between(x, mean_5 - std_5, mean_5 + std_5, color='y', alpha=0.2)
plt.legend(title=r'$\alpha$', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("10-armed Contextual Bandit problem: LinUCB")
plt.show()

#results = open("Softmax_Results", "a")
#np.savetxt(results, mean_2)
#results.close()