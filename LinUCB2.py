#https://github.com/akjayant/News-Article-Recommendation-via-Contextual-Bandits/blob/main/LinUCB.ipynb
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import torch
import pandas as pd
from experiment_weighted_or import MaxLength
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

names = ['action','reward']
names_c = []
for i in range(10):
    for j in range(10):
        names_c.append('context'+str(i)+str(j))
names.extend(names_c)
data = pd.read_csv("dataset.txt", sep=" ", names=names, index_col=False)

print(data.shape)
print(data.action.unique())


def LinUCB(n_arms, context_dim, alpha, data, rounds, train_flag):
    A = np.eye(context_dim)
    A = np.vstack([A] * n_arms).reshape(context_dim, context_dim, n_arms)
    # A = np.zeros([n_arms,context_dim,context_dim])
    b = np.zeros([n_arms, context_dim, 1])
    # print(A.shape,b.shape)
    cum_reward = 0
    reward_history = np.zeros(rounds)
    cum_reward_history = np.zeros(rounds)
    n_update = 0
    print(data.shape[0])
    for i in range(data.shape[0]):
        qa = []
        for j in range(n_arms):
            context_a = np.array(data.loc[i, 'context' + str(j) + '0':'context' + str(j) + '9']).reshape(context_dim, 1)
            #print(i, j, context_a)
            A_inv = np.linalg.inv(A[j])
            theta_a = np.matmul(A_inv, b[j])
            p_ta = np.dot(theta_a.T, context_a) + alpha * np.sqrt(np.matmul(np.matmul(context_a.T, A_inv), context_a))
            qa.append(p_ta[0][0])
            # print(qa)
        # print(qa)
        print(i)
        action = np.random.choice(np.where(qa == max(qa))[0])
        # print(action)
        context_a_selected = np.array(data.loc[i, 'context' + str(action) + '0':'context' + str(action) + '9']).reshape(
            context_dim, 1)
        data_action = data.loc[i, 'action'] - 1
        if data_action == action:
            n_update += 1
            #r = data.loc[i, 'reward']
            c = MaxLength(gym.Wrapper, 50)
            r, collected = c.reward(max_trajectory=50, weight=action / 10)
            A[action] = A[action] + np.outer(context_a_selected, context_a_selected)
            b[action] = b[action] + r * context_a_selected
            reward_history[n_update] = r
            cum_reward += r
            cum_reward_history[n_update] = cum_reward
        if n_update == rounds - 1:
            break

    return reward_history, cum_reward_history

rhistory,chistory = LinUCB(10,10,1,data,1000,True)

plt.plot(chistory[:1000]/np.linspace(1,1000,1000))
plt.show()

