import h5py
import random
import numpy as np
import multiprocessing as mp

from collections import deque
from vec_env_staliro import clear_mpi_env_vars
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

DEBUG = False
def worker_acc(remote, parent_remote, env_fn_wrapper):
    """
    modified worker to call staliro

    :param remote:
    :param parent_remote:
    :param env_fn_wrapper:
    :return:
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            # HighD
            # reset: self.highd_vels[idx], self.highd_accs[idx, :]
            # step: action, self.highd_vels[idx], self.highd_accs[idx, :]
            
            # S-Taliro
            # reset: [staliro_trace_queue]                              % obs_pos, obs_vel, obs_driver
            # step: [action, self.staliro_trace_queue]                  % action, self.obs_pos, self.obs_vel, self.obs_driver

            if cmd == 'step':
                ob, reward, done, info = env.step(data[0])
                if done:
                    info["last_ob"] = ob
                    if len(data) == 3: # HighD
                        ob = env.reset(obs_vel=data[1], obs_driver=data[2])
                    else: # S-Taliro
                        staliro_trace_queue = data[1]
                        staliro_obs_trace = random.choice(staliro_trace_queue)
                        if DEBUG:
                            print("<Debugging worker_acc>:  staliro_obs_trace.shape {}".format(staliro_obs_trace.shape))
                        obs_pos = staliro_obs_trace[0]
                        obs_vel = staliro_obs_trace[1]
                        obs_driver = staliro_obs_trace[2:]
                        if DEBUG:
                            print("<subproc_vec_env_acc/worker_acc>: init_pos {}, init_vel {}".format(obs_pos, obs_vel))
                        # ob = env.reset(obs_x=data[1], obs_vel=data[2], obs_driver=data[3])
                        ob = env.reset(obs_x=obs_pos, obs_vel=obs_vel, obs_driver=obs_driver)
                remote.send((ob, reward, done, info))          

            elif cmd == 'reset':
                if len(data) == 2: # HighD
                    ob = env.reset(obs_vel=data[0], obs_driver=data[1])
                else: # S-Taliro
                    # ob = env.reset(obs_x=data[0], obs_vel=data[1], obs_driver=data[2])
                    staliro_trace_queue = data[0]
                    staliro_obs_trace = random.choice(staliro_trace_queue)
                    if DEBUG:
                        print("<Debugging worker_acc>:  staliro_obs_trace.shape {}".format(staliro_obs_trace.shape))
                    obs_pos = staliro_obs_trace[0]
                    obs_vel = staliro_obs_trace[1]
                    obs_driver = staliro_obs_trace[2:]
                    if DEBUG:
                        print("<subproc_vec_env_acc/worker_acc>: init_pos {}, init_vel {}".format(obs_pos, obs_vel))                    
                    ob = env.reset(obs_x=obs_pos, obs_vel=obs_vel, obs_driver=obs_driver)
                remote.send(ob)
            elif cmd == 'render':
                # remote.send(env.render(mode='rgb_array'))
                env.render()
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space, env.spec))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnvAcc(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, h5_path=None, spaces=None, context='spawn'):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=worker_acc, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()
        self.viewer = None

        # load HighD data
        self.highd_f = h5py.File(h5_path, "r")
        self.highd_vels = self.highd_f['initial_velocities']
        self.highd_accs = self.highd_f['x_accelerations']


        self.staliro_trace_queue = None

        # self.obs_pos = None
        # self.obs_vel = None
        # self.obs_driver = None

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()

        # if self.obs_pos is None or self.obs_vel is None or self.obs_driver is None:
        if self.staliro_trace_queue is None:
            idxs = random.sample(range(len(self.highd_vels)), self.num_envs)
            # print("<SubprocVecEnvACC/step> idxs: {}".format(idxs))
            for remote, action, idx in zip(self.remotes, actions, idxs):
                remote.send(('step', [action, self.highd_vels[idx], self.highd_accs[idx, :]]))
        else:
            for remote, action in zip(self.remotes, actions):
                # remote.send(('step', [action, self.obs_pos, self.obs_vel, self.obs_driver]))
                remote.send(('step', [action, self.staliro_trace_queue]))

        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    # only S-Taliro assigns obs_vel and obs_driver for reset function
    # def reset(self, obs_pos=None, obs_vel=None, obs_driver=None):
    def reset(self, staliro_trace_queue=None):
        self._assert_not_closed()

        # if obs_pos is None or obs_vel is None or obs_driver is None:
        if staliro_trace_queue is None:
            # sample accel and init_velocity from highd
            idxs = random.sample(range(len(self.highd_vels)), self.num_envs)
            # print("<SubprocVecEnvACC/step> idxs: {}".format(idxs))
            for remote, idx in zip(self.remotes, idxs):
                remote.send(('reset', [self.highd_vels[idx], self.highd_accs[idx, :]]))
        else:

            self.staliro_trace_queue = staliro_trace_queue
            # self.obs_pos = obs_pos
            # self.obs_vel = obs_vel
            # self.obs_driver = obs_driver

            for remote in self.remotes:
                remote.send(('reset', [staliro_trace_queue]))
                # remote.send(('reset', [obs_pos, obs_vel, obs_driver]))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        print("Closing_extras...")
        # close highD dataset
        self.highd_f.close()
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def render(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)