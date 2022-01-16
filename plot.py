import os
# import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

FONT_SIZE=13

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    # shape = a.shape[:-1] + (a.shape[-1], window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def plot_curves(x_list, y_lists, xaxis, title, label, color):
    # plt.figure(figsize=(8,2))
    maxx = x_list[-1]
    minx = 0
    x = x_list
    y_mean = np.mean(y_lists, axis=0)
    y_std_error = np.std(y_lists, axis=0) / np.sqrt(y_lists.shape[0])
    y_upper = y_mean + y_std_error
    y_lower = y_mean - y_std_error

    # for (i, (x, y)) in enumerate(xy_list):
    #     color = COLORS[i]
    #     plt.scatter(x, y, s=2)
    x_filter, y_mean = window_func(x, y_mean, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    _, y_upper = window_func(x, y_upper, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    _, y_lower = window_func(x, y_lower, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes

    plt.plot(x_filter, y_mean, label=label, color=color)

    plt.fill_between(x_filter, y_lower, y_upper, alpha=0.25)
    # minx = 10000
    #plt.xlim(minx, maxx)
    #plt.ylim(-210, -120)
    plt.title(title, fontsize=FONT_SIZE)
    plt.xlabel(xaxis, fontsize=FONT_SIZE)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.ylabel("Episode Rewards", fontsize=FONT_SIZE)
    plt.legend()
    # 横坐标指数表示
    # plt.xscale('symlog')
    plt.tight_layout()

# read csv
# path = "/home/mqr/TempoRL-master/experiments/featurized_results/sparsemountain/tdqn/"
# files = os.listdir(path)
# # 测出sample个数
# rew_1 = pd.read_csv(path + files[1])
# x_val = rew_1['Step'].values

# read txt
# path = "/home/mqr/TempoRL-master/experiments/featurized_results/sparsemountain/tdqn/"
# files = os.listdir(path)
# rew_1 = np.loadtxt(path+files[0]+"/reward.txt")
# x_val = np.loadtxt(path+files[0]+"/step.txt")
# len_x = len(x_val)
# # number of files
# len_file = len(files)
# rew = np.array(np.zeros(shape = (len_file, len_x)))
# for file_i in range(len_file):
#     rew[file_i] = pd.read_csv(path+files[file_i])['Value'].values


# todo main
step_1 = np.loadtxt("models/Hopper_v2/GaussianMixture/run34/k=10-steps.txt")
rew_1 = np.loadtxt("models/Hopper_v2/GaussianMixture/run34/k=10-reward.txt")
rew_2 = np.loadtxt("models/Hopper_v2/GaussianMixture/run33/k=20-reward.txt")
step_2 = np.loadtxt("models/Hopper_v2/GaussianMixture/run33/k=20-steps.txt")
rew_3 = np.loadtxt("models/Hopper_v2/GaussianMixture/run4/k=50-reward.txt")
step_3 = np.loadtxt("models/Hopper_v2/GaussianMixture/run4/k=50-steps.txt")
rew_4 = np.loadtxt("models/Hopper_v2/GaussianMixture/run5/k=100-reward.txt")
step_4 = np.loadtxt("models/Hopper_v2/GaussianMixture/run5/k=100-steps.txt")

# x_val = np.loadtxt("/home/qirui/code/pytorch-sac/models/Hopper_v2/GaussianMixture/run2/k=50-steps.txt")
# rew = np.loadtxt("/home/qirui/code/pytorch-sac/models/Hopper_v2/GaussianMixture/run2/k=50-reward.txt")
# gaussian = np.loadtxt("/home/qirui/code/pytorch-sac/models/Hopper_v2/GaussianMixture/run1/k=50-reward.txt")
# gaussian_step = np.loadtxt("/home/qirui/code/pytorch-sac/models/Hopper_v2/GaussianMixture/run1/k=50-steps.txt")

rew_1 = np.expand_dims(rew_1, axis=0)
rew_2 = np.expand_dims(rew_2, axis=0)
rew_3 = np.expand_dims(rew_3, axis=0)
rew_4 = np.expand_dims(rew_4, axis=0)
plot_curves(x_list=step_1, y_lists=rew_1, xaxis="train steps", title="Hopper-v2", label="seed1: SAC + GMM k=%d"%10, color='b')
plot_curves(x_list=step_2, y_lists=rew_2, xaxis="train steps", title="Hopper-v2", label="seed1: SAC + GMM k=%d"%20, color='r')
plot_curves(x_list=step_3, y_lists=rew_3, xaxis="train steps", title="Hopper-v2", label="seed1: SAC + GMM k=%d"%50, color='g')
plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2", label="seed1: SAC + GMM k=%d"%100, color='c')
plt.show()


