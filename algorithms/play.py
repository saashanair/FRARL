""" This script augments the functionality of run.py with a play function"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append(os.path.join(os.getcwd(), '../'))
sys.path.append("./rl_algs")
import h5py
import random
import constants

# import gym_acc
# from gym_car_sim_julia_acc.envs import CarSimEnvACC
# sys.path.append("/home/xiao/projects/mlmeetsformalmethods/simulator_python/gym_acc")
from gym_car_acc.envs import CarACCEnv

import numpy as np
import gym
from model import PlayModel

# importing build_policy from baselines.common.policies causes problem for matlab
# so copied policies in openai folder
from policies import build_policy
# from baselines.common.policies import build_policy

DEBUG = False

def get_logging_path(args, extra_args):
    if  "call_staliro" not in extra_args.keys():
        return os.path.join(constants.LOG_DIR, "-".join((args.env, args.alg)), "Gym-env-test")
    return os.path.join(constants.LOG_DIR, "-".join((args.env, args.alg)), extra_args["call_staliro"])


# traj = play_once(env, model, obs_init_vel, total_sim_time, obs_acc_trace)
def play_once(env, model, obs_s, obs_init_vel, total_sim_time, obs_acc_trace, video=False, reward=False):
    """
    Runs one simulation
    :param env: simulation env
    :param model: model with loaded parameters
    :param obs_s: initial obstacle position
    :param obs_init_vel: initial obstacle velocity
    :param total_sim_time: maximal simulation time steps N
    :param obs_acc_trace: Nx1 matrix with longitudinal acceleration trace of the obstacle
    :param video: whether to render scene
    :return: time sequence of ego observations
    """

    traj = []
    if DEBUG:
        obs = env.reset()
    else:
        obs = env.reset(obs_x=obs_s, obs_driver=obs_acc_trace, obs_vel=obs_init_vel)
        if video:
            camtype = "follow_scene"
            env.render(camtype=camtype)
    done = False
    episode_rew = 0
    step_counter = 0
    while not done and (step_counter < total_sim_time):
        ego_actions, _, _, _ = model.step(obs)
        obs, rew, done, _ = env.step(ego_actions[0])
        if video:
            env.render(camtype=camtype)
        episode_rew += rew

        done = done.any() if isinstance(done, np.ndarray) else done
        traj.append([step_counter] + obs[:])
        step_counter += 1
    if reward:
        return traj, episode_rew
    else:
        return traj


def get_random_trace(fn):

    vels = fn['initial_velocities']
    accels = fn['x_accelerations']
    idx = random.sample(range(len(fn['initial_velocities'])), 1)
    return vels[idx][0], accels[idx, :][0]

def play_n_times(env, model, n, video=False):

    # load HighD data
    highd_f = h5py.File("./highD/HighD_straight.h5", "r")

    vel, accel = get_random_trace(highd_f)
    obs = env.reset(obs_driver=accel, obs_vel=vel)

    if video:
        camtype = "follow_scene"
        env.render(camtype=camtype)

    episode_rew = 0
    rewards = []
    actions = []
    observations = []
    i = 0
    while i < n:
        print("{}/{}".format(i+1, n), end="\r")
        action, _, _, _ = model.step(obs)
        obs, rew, done, _ = env.step(action[0])
        actions.append(action[0])
        observations.append(obs)
        if video:
            env.render(camtype=camtype)
        episode_rew += rew
        # env.render()
        # done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            i += 1
            rewards.append(episode_rew)
            episode_rew = 0

            vel, accel = get_random_trace(highd_f)
            obs = env.reset(obs_driver=accel, obs_vel=vel)
    highd_f.close()

    return rewards, observations, actions

# ================================================================
# Function to evaluate trained models, build env, build model, load model
# ================================================================
# needs those useless args otherwise network_kwargs got extra unknown args
def build_model(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, call_staliro=False, **network_kwargs):

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = 1

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)

    model = PlayModel(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm)

    return model


def build_simple_env(args):
    """ build an unwrapped gym env cause wrapped ones with baselines have import issues in matlab"""
    # TODO: Implement random seed in Julia environment

    env_id = args.env
    assert env_id == "CarSim-v0" or env_id == "CarSim-ACC-v1" or env_id == "CarSim-Python-v1", \
     "Env {} is not supported!".format(env_id)

    if DEBUG:
        env_id = "CartPole-v0"
    if args.seed is not None:
        env = gym.make(env_id, seed=args.seed)
    else:
        env_params = dict()
        env_kwargs = dict(render_params=dict(zoom=2.5, viz_dir="/tmp"))
        env = gym.make(env_id, env_params=env_params, **env_kwargs)

    return env

# py.play.play_sim(XPoint, simTime, InpSignal);
def play_sim(XPoint, total_sim_time, obs_acc_trace):
    """
    Builds simulation env and model, and simulates with given initial obstacle position and total simulation time
    :param XPoint: [initial obstacle position; initial obstacle velocity)
    :param total_sim_time: total simulation time steps
    :param obs_input_trace: lateral and longitudinal acceleration trace of the obstacle
    :return: a time series trajectory of ego observations
    """
    # TODO: right now everytime S-Taliro is called, one new env and one new model are created, this is not efficient.
    # problem: can't transfer an Env class or Model class from python (worker) to matlab (staliro) and to python (play)
    # solution:
    # print("play_sim obs_input_trace shape: {}".format(np.array(obs_input_trace).shape))
    # load args and extra_args
    load_path = constants.LOG_DIR # logger.get_dir(), constants.SAVE_PATH)
    args = np.load(os.path.join(load_path, "args.npz"), allow_pickle=True)["args"].item()
    alg_kwargs = np.load(os.path.join(load_path, "alg_kwargs.npz"), allow_pickle=True)["alg_kwargs"].item()

    # build env
    args.num_env = 1
    env = build_simple_env(args)

    # build model
    model = build_model(
        env=env,
        seed=args.seed,
        total_timesteps=int(args.num_timesteps),
        **alg_kwargs
    )

    # load model
    if not DEBUG:
        load_path = get_logging_path(args, alg_kwargs)
        model.load(load_latest_checkpoint(load_path))

    obs_s = XPoint[0]
    obs_init_vel = XPoint[1]
    # print(obs_s, obs_init_vel)
    traj = play_once(env, model, obs_s, obs_init_vel, total_sim_time, obs_acc_trace)

    return traj


# ================================================================
# End
# ================================================================
def load_latest_checkpoint(rootdir):
    load_path_model = os.path.join(rootdir, "checkpoints")
    files = os.listdir(load_path_model)
    ckpt_path = os.path.join(load_path_model, files[np.argmax([int(x) for x in files])])
    # print("Loading model from {}".format(ckpt_path))
    return ckpt_path

def evaluate(load_path, video=False, N=14):

    # load_path = constants.LOG_DIR  # logger.get_dir(), constants.SAVE_PATH)
    # load_path = "./logs/CarSim-Python-v1-ppo2_rlf/groups/no_staliro-1"
    args = np.load(os.path.join(load_path, "args.npz"), allow_pickle=True)["args"].item()
    alg_kwargs = np.load(os.path.join(load_path, "alg_kwargs.npz"), allow_pickle=True)["alg_kwargs"].item()

    # build env
    args.num_env = 1
    args.seed = None
    env = build_simple_env(args)

    # build model
    model = build_model(
        env=env,
        seed=args.seed,
        total_timesteps=int(args.num_timesteps),
        **alg_kwargs
    )

    # load model
    if not DEBUG:
        # load_path = get_logging_path(args, alg_kwargs)
        model.load(load_latest_checkpoint(load_path))

    rewards, observations, actions = play_n_times(env, model, N, video=video)
    print(np.average(np.array(rewards)))
    return rewards, model, env
    # np.save(os.path.join(load_path, "reward_normal_{}.npy".format(alg_kwargs["call_staliro"])), np.asarray(rewards))
    # observations, actions = play_n_times(env, model, 29, video=True)
    # np.save("observations.npy", np.asarray(observations))
    # np.save("actions.npy", np.asarray(actions))



def evaluate_with_staliro_trace(load_path, video=False, output_reward=False):

    # load model and env
    # load_path = constants.LOG_DIR  # logger.get_dir(), constants.SAVE_PATH)
    args = np.load(os.path.join(load_path, "args.npz"), allow_pickle=True)["args"].item()
    alg_kwargs = np.load(os.path.join(load_path, "alg_kwargs.npz"), allow_pickle=True)["alg_kwargs"].item()

    # build env
    args.num_env = 1
    env = build_simple_env(args)
    args.seed = None
    # build model
    model = build_model(
        env=env,
        seed=args.seed,
        total_timesteps=int(args.num_timesteps),
        **alg_kwargs
    )

    # load model
    if not DEBUG:
        # load_path = get_logging_path(args, alg_kwargs)
        model.load(load_latest_checkpoint(load_path))

    # load staliro traces
    staliro_path = "./logs/CarSim-Python-v1-ppo2_rlf/debug"

    files = os.listdir(staliro_path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    rewards = []
    for file in files:
        trace = np.squeeze(np.load(os.path.join(staliro_path, file)))  # (281,)
        obs_s = trace[0]
        obs_init_vel = trace[1]
        obs_acc_trace = trace[2:]

        total_sim_time = obs_acc_trace.shape[0]
        _, reward = play_once(env, model, obs_s, obs_init_vel, total_sim_time, obs_acc_trace, video=video, reward=output_reward)
        rewards.append(reward)
    print(np.average(np.array(rewards)))
    np.save("{}_eval_staliro_reward.npy".format(alg_kwargs["call_staliro"]), np.asarray(rewards))


def main():

    obs_xs = [500., 400., 300.]
    total_sim_time = 1500
    obs_acc_trace = np.zeros(total_sim_time)
    for obs_x, total_sim_time in obs_xs:
        play_sim(obs_x, total_sim_time, obs_acc_trace)

def test():
    print("Hello from test function in play.py!")

def simple_run():
    obs_s = 60.
    obs_init_vel = 30.
    total_sim_time = 1500
    obs_input_trace = np.zeros(total_sim_time)
    
    XPoint = []
    XPoint.append(obs_s)
    XPoint.append(obs_init_vel)
    traj = play_sim(XPoint, total_sim_time, obs_input_trace)

    obs_acc = []
    for obs in traj:
        obs_acc.append(obs[4])

def evaluate_groups(load_path, video=False, N=14):
    # rewards, model, env = evaluate(load_path, video=video, N=N)
    # np.save(os.path.join(load_path, "../../rewards", "{}_normal.npy".format(os.path.basename(load_path))), np.asarray(rewards))
    #
    args = np.load(os.path.join(load_path, "args.npz"), allow_pickle=True)["args"].item()
    alg_kwargs = np.load(os.path.join(load_path, "alg_kwargs.npz"), allow_pickle=True)["alg_kwargs"].item()
    args.num_env = 1
    args.seed = None
    env = build_simple_env(args)
    model = build_model(env=env, seed=args.seed, total_timesteps=int(args.num_timesteps), **alg_kwargs)
    model.load(load_latest_checkpoint(load_path))
    #
    rewards = evaluate_groups_staliro(model, env, video=video, N=N)
    np.save(os.path.join(load_path, "../../rewards", "{}_staliro.npy".format(os.path.basename(load_path))), np.asarray(rewards))

def evaluate_groups_staliro(model, env, video=False, N=14):
    # load staliro traces
    staliro_path = "./logs/CarSim-Python-v1-ppo2_rlf/evaluation/total_staliro_traces.npy"
    staliro_trace = np.load(staliro_path)

    idxs = random.sample(range(staliro_trace.shape[-1]), N)
    rewards = []
    for i, idx in enumerate(idxs):
        print("{}/{}".format(i+1, N), end="\r")
        trace = staliro_trace[:, idx]        
        obs_s = trace[0]
        obs_init_vel = trace[1]
        obs_acc_trace = trace[2:]
        total_sim_time = obs_acc_trace.shape[0]
        _, reward = play_once(env, model, obs_s, obs_init_vel, total_sim_time, obs_acc_trace, video=video, reward=True)
        rewards.append(reward)
    print(np.average(np.array(rewards)))
    return rewards    

if __name__ == '__main__':      
    # evaluate_with_staliro_trace(video=True, output_reward=True)
    # evaluate(video=True, N=10)
    # test()
    # simple_run()
    # evaluate(video=True, N=15)
    # path = /home/xiao/projects/mlmeetsformalmethods/Implementation/openai/logs/CarSim-Python-v1-ppo2_rlf/
    # evaluation/reward/models/multiple_staliro-1
    evaluate_groups(sys.argv[-1], video=False, N=1000)
    # evaluate_groups_staliro(sys.argv[-1], video=False, N=100)


