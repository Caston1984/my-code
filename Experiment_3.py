
import matplotlib.pyplot as plt
import numpy as np

zoom = open('Zoom_Results',"r")
lines = zoom.readlines()
mean_zoom = []
std_zoom = []
for x in lines:
    mean_zoom.append(float(x.split(' ')[0]))
    std_zoom.append(float((x.split(' ')[2]).split('\n')[0]))
zoom.close()

hoo = open('HOO_Results',"r")
lines = hoo.readlines()
mean_hoo = []
std_hoo = []
for x in lines:
    mean_hoo.append(float(x.split(' ')[0]))
    std_hoo.append(float((x.split(' ')[2]).split('\n')[0]))
hoo.close()


ucb = open('UCB_Results',"r")
lines = ucb.readlines()
mean_ucb = []
std_ucb = []
for x in lines:
    mean_ucb.append(float(x.split(' ')[0]))
    std_ucb.append(float((x.split(' ')[2]).split('\n')[0]))
ucb.close()

sof = open('Softmax_Results',"r")
lines = sof.readlines()
mean_sof = []
std_sof = []
for x in lines:
    mean_sof.append(float(x.split(' ')[0]))
    std_sof.append(float((x.split(' ')[2]).split('\n')[0]))
sof.close()

exp = open('EXP3_Results',"r")
lines = exp.readlines()
mean_exp = []
std_exp = []
for x in lines:
    mean_exp.append(float(x.split(' ')[0]))
    std_exp.append(float((x.split(' ')[2]).split('\n')[0]))
exp.close()

TS = open('TS_Results',"r")
lines = TS.readlines()
mean_TS = []
std_TS = []
for x in lines:
    mean_TS.append(float(x.split(' ')[0]))
    std_TS.append(float((x.split(' ')[2]).split('\n')[0]))
TS.close()

x = np.arange(len(mean_zoom))
plt.plot(x, np.array(mean_zoom), 'b-', label='Zooming')
plt.fill_between(x, np.array(mean_zoom) - np.array(std_zoom), np.array(mean_zoom) + np.array(std_zoom), color='b', alpha=0.2)

plt.plot(x, np.array(mean_hoo), 'r-', label='HOO')
plt.fill_between(x, np.array(mean_hoo) - np.array(std_hoo), np.array(mean_hoo) + np.array(std_hoo), color='r', alpha=0.2)

plt.plot(x, np.array(mean_ucb), 'g-', label='UCB')
plt.fill_between(x, np.array(mean_ucb) - np.array(std_ucb), np.array(mean_ucb) + np.array(std_ucb), color='g', alpha=0.2)

plt.plot(x, np.array(mean_sof), 'k-', label='Softmax')
plt.fill_between(x, np.array(mean_sof) - np.array(std_sof), np.array(mean_sof) + np.array(std_sof), color='k', alpha=0.2)

plt.plot(x, np.array(mean_exp), 'y-', label='Exp3')
plt.fill_between(x, np.array(mean_exp) - np.array(std_exp), np.array(mean_exp) + np.array(std_exp), color='y', alpha=0.2)

plt.plot(x, np.array(mean_TS), 'm-', label='Thompson Sampling')
plt.fill_between(x, np.array(mean_TS) - np.array(std_TS), np.array(mean_TS) + np.array(std_TS), color='m', alpha=0.2)

plt.ylim(-6, -3)
plt.legend(title='Bandit Algorithms', loc='best')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Comparison of the Bandit Algorithms")
plt.show()
