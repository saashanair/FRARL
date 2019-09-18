import numpy as np
from baselines.common.runners import AbstractEnvRunner

from rl_algs.staliro_experiment_generator import get_staliro_experiment_builder

from collections import deque

class RunnerStaliro(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        # self.staliro_trace_queue = deque(maxlen=10) # TODO: Why 5?
        self.staliro_trace_queue = deque()

    def run(self, data, call_staliro):
        staliro_called = False
        obs_acc_trace = None
        # staliro_trace_queue.shape = (1502, 1)
        staliro_trace_queue, staliro_robustness_value = get_staliro_experiment_builder(call_staliro)(data, self.staliro_trace_queue)
        if staliro_trace_queue != None:
            staliro_called = True
            self.staliro_trace_queue = staliro_trace_queue
            prob = None
            if call_staliro == "weighted_queue":
                num_old_el_in_queue = len(staliro_trace_queue) - 1
                prob = [0.1] * num_old_el_in_queue + [1.0 - (num_old_el_in_queue / 10)]            

            self.obs[:] = self.env.reset(staliro_trace_queue=staliro_trace_queue)
            # idx = np.random.choice(len(staliro_trace_queue), p=prob)
            # staliro_obs_trace = staliro_trace_queue[idx]
            # obs_pos = staliro_obs_trace[0, 0]
            # obs_vel = staliro_obs_trace[1, 0]
            # obs_driver = staliro_obs_trace[2:, 0]
            # self.obs[:] = self.env.reset(obs_pos=obs_pos, obs_vel=obs_vel, obs_driver=obs_driver)
        
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        lastobinfos = []
        
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                if isinstance(info, tuple):
                    info = info[0]
                maybeepinfo = info.get('episode')
                epcrashinfo = info.get('last_ob')
                if maybeepinfo: epinfos.append(maybeepinfo)
                if epcrashinfo is not None: lastobinfos.append(epcrashinfo[-2:])

            mb_rewards.append(rewards)
           
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos, lastobinfos, staliro_called)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


