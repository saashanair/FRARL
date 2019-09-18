import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
sys.path.append("./rl_algs")

import constants
# import gym_acc
from gym_car_acc.envs import CarACCEnv

# from gym_car_sim_julia_acc.envs import CarSimEnvACC

import re
import time
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

import logging
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.ERROR)
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_env, make_vec_env
from baselines.bench import Monitor
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)

    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)
    env = build_env(args, eval=False)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    # save args in experiments (for logging purpose) and root folder (for play_sim)
    np.savez(osp.join(constants.LOG_DIR, 'args'), args=args)
    np.savez(osp.join(logger.get_dir(), 'args'), args=args)

    # save alg_kwargs in experiments and root folder
    np.savez(osp.join(constants.LOG_DIR, 'alg_kwargs'), alg_kwargs=alg_kwargs)
    np.savez(osp.join(logger.get_dir(), 'alg_kwargs'), alg_kwargs=alg_kwargs)

    if args.play:
        eval_env = build_env(args, eval=True)
    else:
        eval_env = None

    # print("<run/train>: env {}; eval_env {}".format(env.highd_f.filename, eval_env.highd_f.filename))
    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        **alg_kwargs
    )

    return model, env

# =====================================================================================
# modified make_vec_env from cmd_util.py to make SubprocVecEnvStaliro
# instead of SubprocVecEnv envs
# =====================================================================================
def make_env_staliro(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, logger_dir=None):
    wrapper_kwargs = wrapper_kwargs or {}

    env_params = dict()
    env_kwargs = dict(render_params=dict(zoom=2.5, viz_dir="/tmp/env_{}".format(subrank)))
    env = gym.make(env_id, env_params=env_params, **env_kwargs)

    # env = gym.make(env_id) # subrank

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)
    return env

def make_vec_env_staliro(
        env_id, env_type, num_env, seed,
         wrapper_kwargs=None,
         start_index=0,
         reward_scale=1.0,
         flatten_dict_observations=True,
         gamestate=None,
         eval=False):
    """
    Modified make_vec_env function from cmd_util.py
    to make SubprocVecEnvStaliro instead of SubprocVecEnv envs
    """

    from baselines.common import set_global_seeds
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    # from subproc_vec_env_staliro import SubprocVecEnvStaliro
    from subproc_vec_env_acc import SubprocVecEnvAcc

    wrapper_kwargs = wrapper_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()

    def make_thunk(rank):
        return lambda: make_env_staliro(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            logger_dir=logger_dir
        )

    set_global_seeds(seed)
    h5_path = "./highD/HighD_straight.h5"
    if num_env > 1:
        return SubprocVecEnvAcc([make_thunk(i + start_index) for i in range(num_env)], h5_path)
    else:
        return DummyVecEnv([make_thunk(start_index)])


def build_env(args, eval=False):
    ncpu = multiprocessing.cpu_count()
    nenv = args.num_env or ncpu
    alg = args.alg

    env_type, env_id = get_env_type(args)

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                      intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    flatten_dict_observations = alg not in {'her'}

    if not eval:
        start_index = 0
    else:
        start_index = args.num_env or 0
    env = make_vec_env_staliro(
        env_id,
        env_type,
        nenv or 1,
        args.seed,
        start_index=start_index,
        reward_scale=args.reward_scale,
        flatten_dict_observations=flatten_dict_observations,
        eval=eval
    )

    return env


def get_env_type(args):
    env_id = args.env
    env_type = "envs"

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}

def get_loading_path(args, extra_args):
    from play import load_latest_checkpoint
    return load_latest_checkpoint(get_logging_path(args, extra_args))


def get_logging_path(args, extra_args):
    if  "call_staliro" not in extra_args.keys():
        return os.path.join(constants.LOG_DIR, "-".join((args.env, args.alg)), "Gym-env-test")
    return os.path.join(constants.LOG_DIR, "-".join((args.env, args.alg)), extra_args["call_staliro"])

def main(args):

    start_time = time.time()
    # TODO: restore model and return training
    # load model is simple, but restore loggings will be more difficult
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if "load_path" in extra_args:
        if extra_args["load_path"] is True:
            extra_args["load_path"] = get_loading_path(args, extra_args)
    # ./logs / env + alg / experiments_name
    # [no_staliro, fixed_staliro, randomized_staliro, weighted_queue, variable_start, success_counter]
    os.environ["OPENAI_LOGDIR"] = get_logging_path(args, extra_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        model.save(osp.join(logger.get_dir(), args.save_path, 'model.pkl'))

    print("Elapsed time {}".format(time.time() - start_time))
    return model

if __name__ == '__main__':
    main(sys.argv)
