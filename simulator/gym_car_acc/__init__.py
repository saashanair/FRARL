from gym.envs.registration import register

register(
    id='CarSim-Python-v1',
    entry_point='gym_car_acc.envs:CarACCEnv'
)
