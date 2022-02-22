
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ucb = np.loadtxt("UCB_Results")
exp3 = np.loadtxt("EXP3_Results")
softmax = np.loadtxt("Softmax_Results")
TS = np.loadtxt("TS_Results")



x = np.arange(len(ucb))
plt.plot(x, ucb, 'b-', label='UCB')
#plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, exp3, 'r-', label='EXP 3')
#plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.plot(x, softmax, 'k-', label='Softmax')
#plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='k', alpha=0.2)
plt.plot(x, TS, 'g-', label='Thompson Sampling')
#plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='g', alpha=0.2)
plt.legend(title='MAB Algorithms', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Comparison of the MAB Algorithms")
plt.show()

