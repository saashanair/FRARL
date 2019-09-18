import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.prediction.prediction import TrajectoryPrediction

from gym_car_acc.envs import params

def gen_straight_lane(lane_length=params.DEFAULT_VEHICLE_LENGTH, lane_width=params.DEFAULT_VEHICLE_WIDTH):
    """
    Generates lanelet network with one straight lanelet.
    :param lane_length: length of straight lane
    :param lane_width: width of straight lane
    :return: LaneletNetwork contains one straight lanelet
    """
    left_coord = lane_width / 2.
    right_coord = -lane_width / 2.

    left_vertices = [[0., left_coord], [lane_length, left_coord]]
    right_vertices = [[0., right_coord], [lane_length, right_coord]]
    center_vertices = [[0., 0.], [lane_length, 0.]]

    lanelet = Lanelet(
        lanelet_id=1101,
        left_vertices=np.array(left_vertices),
        right_vertices=np.array(right_vertices),
        center_vertices=np.array(center_vertices)
    )

    lanelet_network = LaneletNetwork()
    lanelet_network.add_lanelet(lanelet)

    return lanelet_network

def gen_vehicle(vehicle_id=None, init_position=None, init_velocity=None):
    """
    Generates obstacle with given id, initial position and initial velocity;
    Default obstacle type: CAR, default obstacle shape: Rectangle
    :param vehicle_id:
    :return: DynamicObstacle
    """

    assert all(v is not None for v in [vehicle_id, init_velocity, init_position]), \
        '<utils/gen_vehicle> vehicle id, initial position and initial velocity need to be given'

    obstacle_shape = Rectangle(length=params.DEFAULT_VEHICLE_LENGTH, width=params.DEFAULT_VEHICLE_WIDTH)
    initial_state = State(
        position=np.array([init_position, 0.]), velocity=init_velocity, orientation=0., acceleration=0., time_step=1
    )

    trajectory = Trajectory(state_list=[initial_state], initial_time_step=1)
    prediction = TrajectoryPrediction(trajectory, obstacle_shape)

    return DynamicObstacle(
        obstacle_id=vehicle_id,
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=obstacle_shape,
        initial_state=initial_state,
        prediction=prediction
    )

def remove_vehicles_from_scenario(scenario: Scenario, vehicle_id: int):

    vehicle = scenario.obstacle_by_id(vehicle_id)
    if isinstance(vehicle, DynamicObstacle):
        scenario.remove_obstacle(vehicle)
    return scenario

def pop(array: np.array):
    return array[0], array[1:]

def get_draw_params(ego=True):

    if ego:
        color = "#d95558"
    else:
        color = "#1d7eea"

    return {
        'occupancy': {
            'draw_occupancies': 1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
                'shape': {
                    'rectangle': {
                       'opacity': 1.0,
                       'facecolor': color,
                       'edgecolor': color,
                       'linewidth': 0.5,
                       'zorder': 18,
                    }
                },
        }
    }
