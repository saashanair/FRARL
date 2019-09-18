from baselines.common import plot_util as pu
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use('PS')

import sys

group_path = sys.argv[-2]

img_path =sys.argv[-1]

plt.figure(figsize=(20, 20))
# plt.rc('text', usetex=True)
# path = "./logs/CarSim-Python-v1-ppo2_rlf/"
results = pu.load_results(group_path)

f, axarr = pu.plot_results(results, average_group=True, split_fn=lambda _: ' ')

plt.xlabel("Training step") # training steps , labelpad=10
plt.ylabel("Reward") # reward
plt.savefig(img_path)
# plt.show()
