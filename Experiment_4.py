
import matplotlib.pyplot as plt
import numpy as np

linUCB = open('LinUCB_Results',"r")
lines = linUCB.readlines()
mean_linUCB = []
std_linUCB = []
for x in lines:
    mean_linUCB.append(float(x.split(' ')[0]))
    std_linUCB.append(float((x.split(' ')[2]).split('\n')[0]))
linUCB.close()

linTS = open('LinTS_Results',"r")
lines = linTS.readlines()
mean_linTS = []
std_linTS = []
for x in lines:
    mean_linTS.append(float(x.split(' ')[0]))
    std_linTS.append(float((x.split(' ')[2]).split('\n')[0]))
linTS.close()


x = np.arange(len(mean_linUCB))
plt.plot(x, np.array(mean_linUCB), 'b-', label='LinUCB')
plt.fill_between(x, np.array(mean_linUCB) - np.array(std_linUCB), np.array(mean_linUCB) + np.array(std_linUCB), color='b', alpha=0.2)

plt.plot(x, np.array(mean_linTS), 'r-', label='LinThompson Sampling')
plt.fill_between(x, np.array(mean_linTS) - np.array(std_linTS), np.array(mean_linTS) + np.array(std_linTS), color='r', alpha=0.2)


plt.ylim(-10, -4)
plt.legend(title='Bandit Algorithms', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Comparison of the Contextual Bandit Algorithms")
plt.show()
