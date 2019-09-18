import os
import gym
import h5py
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import gym_car_acc
from gym_car_acc.envs import CarACCEnv, params
from commonroad.visualization.draw_dispatch_cr import draw_object


def test_make():
    env = gym.make("CarSim-Python-v1")
    print("observation space")
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.observation_space.high)

    print("action space")
    print(env.action_space)
    print(env.action_space.low)
    print(env.action_space.high)

    env.render(mode="human")


def test_reset():
    env = gym.make("CarSim-Python-v1")
    obs = env.reset(obs_driver=np.zeros(10), obs_vel=4.198945720864373, obs_x=105.80134294276883)
    print("Initial observations: {}".format(obs))

    plot_limits = [-10., params.DEFAULT_LANE_LENGTH + 10., -params.DEFAULT_LANE_WIDTH, params.DEFAULT_LANE_WIDTH]
    plt.figure(figsize=(25, 10))
    draw_object(env.scenario, draw_params={'time_begin': 1}, plot_limits=plot_limits)
    plt.gca().set_aspect('equal')
    plt.show()


def test_step():
    env = gym.make("CarSim-Python-v1")
    env.reset(obs_driver=np.zeros(1000), obs_vel=20., obs_x=50.)

    for i in range(120):
        features, rewards, done, infos = env.step(np.array([-1.5]))
        print("Step {}: obs {}, reward {}, done {}, infos {}".format(i, features, rewards, done, infos))


def test_step_timer():
    env = gym.make("CarSim-Python-v1")
    obs_driver = np.zeros(1000)
    env.reset(obs_driver=obs_driver, obs_vel=20.)

    start_time = time.time()

    i = 0
    while i < 3e5:
        features, rewards, done, infos = env.step(np.array([1.5]))
        i += 1
        if done:
            env.reset(obs_driver=obs_driver, obs_vel=20.)
    print("Elapsed time: {}".format(time.time() - start_time))
           

def test_render():
    env = gym.make("CarSim-Python-v1", viz_dir="./imgs")
    obs_driver = np.zeros(1000)
    env.reset(obs_driver=obs_driver, obs_vel=67., obs_x=15.)

    for i in range(10):
        features, rewards, done, infos = env.step(np.array([-1.55555]))
        # if i > 40:
        env.render()
        print("Step {}: obs {}, reward {}, done {}, infos {}".format(i, features, rewards, done, infos))


def test_step_highd():
    env = gym.make("CarSim-Python-v1")#, viz_dir="./imgs")
    h5_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../algorithms/highD/HighD_straight.h5")
    f = h5py.File(h5_path, "r")
    obs_vels = f["initial_velocities"]
    obs_accels = f["x_accelerations"]

    idx = random.sample(range(len(obs_vels)), 1)[0]
    init_obs = env.reset(obs_x=50., obs_driver=obs_accels[idx, :], obs_vel=obs_vels[idx])
    # env.render()
    print("Initial obs: ", init_obs)

    done = False
    i = 1

    while not done:
        ego_action = np.array([-5.])
        obs, reward, done, info = env.step(ego_action)
        # env.render()
        print("step {0}: action {1}, features {2}; reward {3}; done {4}".format(i, ego_action, obs,
                                                                                round(reward, 3), done))
        i += 1

def test_seed(seed=None):
    env = gym.make("CarSim-Python-v1")
    env.seed(seed)
    for _ in range(5):
        obs = env.reset(obs_driver=np.zeros(10), obs_vel=20.)
        print(obs)

if __name__ == "__main__":
    test_step_highd()
