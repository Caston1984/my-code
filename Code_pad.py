import pandas as pd
import numpy as np
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

a = np.array(arm_list).reshape(-1, 1)
r = np.array(reward_list).reshape(-1, 1)
f = np.array(features_list).reshape(-1, 100)

df = pd.DataFrame(a, r)


print(df.head(10))




