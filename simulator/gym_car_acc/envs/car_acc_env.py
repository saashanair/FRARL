import os
# import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
# from gym import error, spaces, utils
# from gym.utils import seeding

from gym_car_acc.envs import utils, params

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.prediction.prediction import Occupancy
from commonroad.visualization.draw_dispatch_cr import draw_object

# from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_object

class CarACCEnv(gym.Env):
    """
    Description:
        One lane environment to simulate Adaptive Cruise Control with one leading vehicle (one-dimensional)

    Observation:
        Distance (ego - obs):               [-DEFAULT_LANE_LENGTH,            0.]
        Relative velocity (ego - obs):      [-math.inf,                 math.inf]
        Absolute ego velocity:              [0.,                        math.inf]
        Obstacle acceleration:              [MIN_ACCEL,                MAX_ACCEL]
        Is colliding:                       [0.,                              1.]
        Ego desired velocity                [EGO_DESIRED_VEL,    EGO_DESIRED_VEL]

    Action:
        Ego acceleration
    """

    def __init__(self, env_id="OneLaneEnv", **kwargs):
        """
        Initializes the environment: defines scenario, generates lanelet network
        :param env_id:
        :param kwargs:
        """
        #
        self._action_space = gym.spaces.Box(low=params.MIN_ACCEL, high=params.MAX_ACCEL, shape=(1,))
        obs_low = np.array([-params.DEFAULT_LANE_LENGTH, -np.inf, 0., params.MIN_ACCEL, 0.]) #, params.EGO_DESIRED_VEL
        obs_high = np.array([0., np.inf, np.inf, params.MAX_ACCEL, 1.]) # , params.EGO_DESIRED_VEL
        self._observation_space = gym.spaces.Box(low=obs_low, high=obs_high) 

        # define scenario and lanelet network
        self.scenario = Scenario(dt=0.04, benchmark_id='ACCEnv')
        lanelet_network = utils.gen_straight_lane(
            lane_width=params.DEFAULT_LANE_WIDTH,
            lane_length=params.DEFAULT_LANE_LENGTH
        )
        self.scenario.add_objects(lanelet_network)

        # define ego and obstacle id
        self.ego_id = 1
        self.obs_id = 2

        # define time step of scenario
        self.epid = 0
        self.t = 0

        # define obstacle accel trace
        self.obs_driver = None

        # define render parameters
        self.plot_limits = [-10., params.DEFAULT_LANE_LENGTH + 10., -params.DEFAULT_LANE_WIDTH, params.DEFAULT_LANE_WIDTH]
        self.viz_dir = kwargs.get("viz_dir", "/tmp")
        if not os.path.exists(self.viz_dir):
            os.mkdir(self.viz_dir)

        # store obstacle and ego vehicle
        self.ego_veh = None
        self.obs_veh = None
        self.ego_state = None
        self.obs_state = None

    def seed(self, seed=None):
    	random.seed(seed)
    	
    def reset(self,
              obs_driver: np.array,
              obs_vel: float,
              obs_x: Union[None, float] = None
              ):
        assert all(v is not None for v in [obs_vel, obs_driver]), \
            '<CarACCEnv/reset> obstacle initial velocity and acceleration trace have to be given'

        # set obstacle acceleration trace
        self.obs_driver = obs_driver

        # reset time step
        self.epid += 1
        self.t = 1

        # remove current vehicles from scenario
        if len(self.scenario.obstacles) > 0:
            self.scenario = utils.remove_vehicles_from_scenario(self.scenario, self.ego_id)
            self.scenario = utils.remove_vehicles_from_scenario(self.scenario, self.obs_id)

        # define ego vehicle and obstacle
        ego_s = 10.
        ego_vel = 65.
        s_safe = (obs_vel ** 2 - ego_vel ** 2)/(2 * params.MIN_ACCEL) + params.DEFAULT_VEHICLE_LENGTH
        if obs_x is None:
            obs_x = np.min([random.random() * 100. + s_safe + ego_s, params.DEFAULT_LANE_LENGTH])
        else:
            obs_x = np.max([ego_s + s_safe, obs_x])
        # add ego and obstacle to scenario
        ego_vehicle = utils.gen_vehicle(vehicle_id=self.ego_id, init_position=ego_s, init_velocity=ego_vel)
        obs_vehicle = utils.gen_vehicle(vehicle_id=self.obs_id, init_position=obs_x, init_velocity=obs_vel)
        self.scenario.add_objects(ego_vehicle)
        self.scenario.add_objects(obs_vehicle)

        return self._get_features()

    def step(self, ego_actions, **kwargs):
        ego_action = ego_actions[0]
        assert isinstance(ego_action, np.float32) or isinstance(ego_action, np.float64), \
             "<CarACCEnv/step> provide action is not float but {}".format(type(ego_action))
        # propagate time step
        self.t += 1

        # read obstacle acceleration
        obs_action, self.obs_driver = utils.pop(self.obs_driver)

        # propagate ego and obstacle state
        for (a, veh_id) in zip([ego_action, obs_action], [self.ego_id, self.obs_id]):
            self._propagate(veh_id, a)

        # return features
        features = self._get_features()
        rewards, done, infos = self._extract_reward(features)

        return (features, rewards, done, infos)

    def _propagate(self, veh_id: int, a: float):

        time_step = self.t
        dt = self.scenario.dt
        obstacle = self.scenario.obstacle_by_id(veh_id)

        last_state = obstacle.prediction.trajectory.state_list[-1] # state_at_time_step(time_step - 1) # #.
        s = last_state.position[0]
        v = last_state.velocity

        next_s = s + v * dt + a * dt * dt / 2
        next_v = v + a * dt

        next_state = State(position=np.array([next_s, 0.]), velocity=next_v, orientation=0., time_step=time_step, acceleration=a)

        # obstacle_shape = obstacle.prediction.shape
        # occupied_region = obstacle_shape.rotate_translate_local(next_state.position, next_state.orientation)
        self.scenario.obstacle_by_id(veh_id).prediction.trajectory.state_list.append(next_state)
        # self.scenario.obstacle_by_id(veh_id).prediction.occupancy_set.append(Occupancy(time_step=time_step, shape=occupied_region))

    # def _check_collision(self):
        # co_obs = create_collision_object(self.scenario.obstacle_by_id(self.obs_id))
        # co_ego = create_collision_object(self.scenario.obstacle_by_id(self.ego_id))
        # return co_obs.collide(co_ego)

    def _get_features(self):

        ego_veh = self.ego_veh = self.scenario.obstacle_by_id(self.ego_id)
        obs_veh = self.obs_veh = self.scenario.obstacle_by_id(self.obs_id)

        ego_state = self.ego_state = ego_veh.prediction.trajectory.state_at_time_step(self.t)
        obs_state = self.obs_state = obs_veh.prediction.trajectory.state_at_time_step(self.t)

        dist = ego_state.position[0] - obs_state.position[0] + 0.5 * (ego_veh.obstacle_shape.length + obs_veh.obstacle_shape.length)
        rel_vel = ego_state.velocity - obs_state.velocity
        ego_vel = ego_state.velocity
        obs_accel = obs_state.acceleration
        is_colliding = float(dist>0.) # self._check_collision()
        # ego_des_vel = params.EGO_DESIRED_VEL
        max_acc = params.MIN_ACCEL

        # return [dist, rel_vel, ego_vel, obs_accel, is_colliding, max_acc]
        return [dist, rel_vel, ego_vel, obs_accel, is_colliding]

    def _extract_reward(self, features):

        keep_alive_reward = 0.1
        goal_reaching_reward = 1.
        crash_reward = -10.

        rewards = 0.
        done = False
        infos = dict()

        is_colliding = features[4]
        ego_vel = self.ego_state.velocity
        obs_vel = self.obs_state.velocity

        obs_veh = self.obs_veh
        obs_state = self.obs_state

        if is_colliding:
            rewards += crash_reward
            done = True
            infos['is_colliding'] = True
        elif ego_vel < 0.:
            rewards += crash_reward
            infos['negative_velocity'] = True
            done = True
        elif self.t == 1500:
            rewards += goal_reaching_reward
            infos['max_timesteps_reaching'] = True
            done = True
        elif obs_state.position[0] + 0.5 * obs_veh.obstacle_shape.length + \
                obs_state.velocity * self.scenario.dt > params.DEFAULT_LANE_LENGTH:
            rewards += goal_reaching_reward
            infos['goal_reaching'] = True
            done = True
        else:
            s_safe = np.min([(obs_vel ** 2 - ego_vel ** 2) / (2 * params.MIN_ACCEL), 1.])
            if abs(features[0]) <= s_safe:
                rewards += 0.1 * crash_reward * np.exp(-5. / s_safe * abs(features[0]))
            # elif ego_vel < obs_vel:
            # 	rewards += 0.05 * crash_reward * np.exp(-5. * ego_vel / obs_vel)
            else:
                rewards += keep_alive_reward

        return rewards, done, infos

    def render(self, mode="rgb array", *args, **kwargs):
        # occupancies are not calculated in step function for efficiency, but calculated here for visualization purpose  

        obs_state = self.obs_state
        ego_state = self.ego_state

        obs_occupancy = Occupancy(
            time_step=1,
            shape=self.obs_veh.prediction.shape.rotate_translate_local(obs_state.position, obs_state.orientation)
            )
        ego_occupancy = Occupancy(
            time_step=1,
            shape=self.ego_veh.prediction.shape.rotate_translate_local(ego_state.position, ego_state.orientation)
            )

        plt.figure(figsize=(25, 10))
        x = ego_state.position[0] - 15.
        y = ego_state.position[1]
        plt.text(x, y+7., "vel={}".format(np.round(ego_state.velocity, 2)))
        accel = str(np.round(ego_state.acceleration, 2))
        plt.text(x, y+14., "accel={}".format(accel))

        x = obs_state.position[0] + 15.
        y = obs_state.position[1]
        plt.text(x, y+7., "vel={}".format(np.round(obs_state.velocity, 2)))
        plt.text(x, y+14., "accel={}".format(np.round(obs_state.acceleration, 2)))
        s_safe = abs((obs_state.velocity ** 2 - ego_state.velocity ** 2)/(2 * params.MIN_ACCEL))
        x = (ego_state.position[0]+obs_state.position[0])/2.

        plt.text(x, y+28., "safe distance {}".format(np.round(s_safe)))
        plt.text(x, y+21., "distance {}".format(np.round(obs_state.position[0]-ego_state.position[0], 2)))

        draw_object(self.scenario.lanelet_network, draw_params={'time_begin': self.t}, plot_limits=self.plot_limits)        
        draw_object(ego_occupancy, draw_params=utils.get_draw_params(ego=True), plot_limits=self.plot_limits)
        draw_object(obs_occupancy, draw_params=utils.get_draw_params(ego=False), plot_limits=self.plot_limits)
        plt.gca().set_aspect('equal')
        if mode == "human":
            plt.show()
        else:
            viz_path = os.path.join(self.viz_dir, "episode_{}".format(self.epid))
            if not os.path.exists(viz_path):
                os.mkdir(viz_path)
            figpath = os.path.join(viz_path, "step_{}.png".format(self.t))
            plt.savefig(figpath)
            print("Rendering saved in {}".format(figpath))
            plt.close()
        # return self.j.render(self.env, *args, **kwargs)

    def obs_names(self):
        pass
        # return self.j.obs_names(self.env)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


def build_space(shape, space_type, info={}):
    if space_type == 'Box':
        if 'low' in info and 'high' in info:
            low = info['low']
            high = info['high']
            if isinstance(info['low'], float):
                low = np.array([low])
                high = np.array([high])
            msg = 'shape = {}\tlow.shape = {}\thigh.shape={}'.format(
                shape, low.shape, high.shape)
            assert shape == np.shape(low) and shape == np.shape(high), msg
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
    elif space_type == 'Discrete':
        assert len(shape) == 1, 'invalid shape for Discrete space'
        return gym.spaces.Discrete(shape[0])
    else:
        raise (ValueError('space type not implemented: {}'.format(space_type)))