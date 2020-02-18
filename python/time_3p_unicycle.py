################################################################################
#
# Script to run a 3 player collision avoidance example
#
################################################################################

from timeit import default_timer as timer

import os
import numpy as np
import matplotlib.pyplot as plt

from unicycle_4d import Unicycle4D
from point_mass_2d import PointMass2D
from product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem

from point import Point

from ilq_solver import ILQSolver
from proximity_cost import ProximityCost
from product_state_proximity_cost import ProductStateProximityCost
from semiquadratic_cost import SemiquadraticCost
from quadratic_cost import QuadraticCost
from player_cost import PlayerCost

from visualizer import Visualizer
from logger import Logger

# General parameters.
TIME_HORIZON = 10.0   # s
TIME_RESOLUTION = 0.1 # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
LOG_DIRECTORY = "./logs/three_player/"

# Create dynamics.
car1 = Unicycle4D()
car2 = Unicycle4D()
car3 = Unicycle4D()
dynamics = ProductMultiPlayerDynamicalSystem(
    [car1, car2, car3], T=TIME_RESOLUTION)

car1_x0 = np.array([
    [-3.0],
    [0.0],
    [0.0],
    [0.05]
])

car2_x0 = np.array([
    [-0.1],
    [3.0],
    [-np.pi / 2.0],
    [0.1]
])

car3_x0 = np.array([
    [0.1],
    [-3.0],
    [np.pi / 2.0],
    [0.1]
])

stacked_x0 = np.concatenate([car1_x0, car2_x0, car3_x0], axis=0)

car1_Ps = [np.zeros((car1._u_dim, dynamics._x_dim))] * HORIZON_STEPS
car2_Ps = [np.zeros((car2._u_dim, dynamics._x_dim))] * HORIZON_STEPS
car3_Ps = [np.zeros((car3._u_dim, dynamics._x_dim))] * HORIZON_STEPS

car1_alphas = [np.zeros((car1._u_dim, 1))] * HORIZON_STEPS
car2_alphas = [np.zeros((car2._u_dim, 1))] * HORIZON_STEPS
car3_alphas = [np.zeros((car3._u_dim, 1))] * HORIZON_STEPS

# Create environment.
car1_position_indices_in_product_state = (0, 1)
car1_goal = Point(3.0, 0.0)
car1_goal_cost = ProximityCost(
    car1_position_indices_in_product_state, car1_goal, np.inf, "car1_goal")

car2_position_indices_in_product_state = (4, 5)
car2_goal = Point(-0.1, -3.0)
car2_goal_cost = ProximityCost(
    car2_position_indices_in_product_state, car2_goal, np.inf, "car2_goal")

car3_position_indices_in_product_state = (8, 9)
car3_goal = Point(0.1, 3.0)
car3_goal_cost = ProximityCost(
    car3_position_indices_in_product_state, car3_goal, np.inf, "car3_goal")

# Penalize speed above a threshold for all players.
car1_v_index_in_product_state = 3
car1_maxv = 2.0  # m/s
car1_maxv_cost = SemiquadraticCost(
    car1_v_index_in_product_state, car1_maxv, True, "car1_maxv")
car1_v_cost = QuadraticCost(car1_v_index_in_product_state, 0.0, "car1_v_cost")

car2_v_index_in_product_state = 7
car2_maxv = 2.0 # m/s
car2_maxv_cost = SemiquadraticCost(
    car2_v_index_in_product_state, car2_maxv, True, "car2_maxv")
car2_v_cost = QuadraticCost(car2_v_index_in_product_state, 0.0, "car2_v_cost")

car3_v_index_in_product_state = 11
car3_maxv = 2.0 # m/s
car3_maxv_cost = SemiquadraticCost(
    car3_v_index_in_product_state, car2_maxv, True, "car2_maxv")
car3_v_cost = QuadraticCost(car3_v_index_in_product_state, 0.0, "car3_v_cost")

# Control costs for all players.
car1_w_cost = QuadraticCost(0, 0.0, "car1_w_cost")
car1_a_cost = QuadraticCost(1, 0.0, "car1_a_cost")

car2_w_cost = QuadraticCost(0, 0.0, "car2_w_cost")
car2_a_cost = QuadraticCost(1, 0.0, "car2_a_cost")

car3_w_cost = QuadraticCost(0, 0.0, "car3_w_cost")
car3_a_cost = QuadraticCost(1, 0.0, "car3_a_cost")

# Proximity cost.
PROXIMITY_THRESHOLD = 1.2
proximity_cost = ProductStateProximityCost(
    [car1_position_indices_in_product_state,
     car2_position_indices_in_product_state,
     car3_position_indices_in_product_state],
    PROXIMITY_THRESHOLD,
    "proximity")

# Build up total costs for both players. This is basically a zero-sum game.
car1_cost = PlayerCost()
car1_cost.add_cost(car1_goal_cost, "x", -300.0)
car1_cost.add_cost(car1_maxv_cost, "x", 50.0)
car1_cost.add_cost(proximity_cost, "x", 50.0)
car1_cost.add_cost(car1_v_cost, "x", 30)
car1_player_id = 0
car1_cost.add_cost(car1_w_cost, car1_player_id, 10.0)
car1_cost.add_cost(car1_a_cost, car1_player_id, 10.0)

car2_cost = PlayerCost()
car2_cost.add_cost(car2_goal_cost, "x", -300.0)
car2_cost.add_cost(car2_maxv_cost, "x", 50.0)
car2_cost.add_cost(proximity_cost, "x", 50.0)
car2_cost.add_cost(car2_v_cost, "x", 30)
car2_player_id = 1
car2_cost.add_cost(car2_w_cost, car2_player_id, 10.0)
car2_cost.add_cost(car2_a_cost, car2_player_id, 10.0)

car3_cost = PlayerCost()
car3_cost.add_cost(car3_goal_cost, "x", -300.0)
car3_cost.add_cost(car3_maxv_cost, "x", 50.0)
car3_cost.add_cost(proximity_cost, "x", 50.0)
car3_cost.add_cost(car3_v_cost, "x", 30)
car3_player_id = 2
car3_cost.add_cost(car3_w_cost, car3_player_id, 10.0)
car3_cost.add_cost(car3_a_cost, car3_player_id, 10.0)

# Visualizer.
visualizer = Visualizer(
    [car1_position_indices_in_product_state,
     car2_position_indices_in_product_state,
     car3_position_indices_in_product_state],
    [car1_goal_cost,
     car2_goal_cost,
     car3_goal_cost],
    [".-r", ".-g", ".-b"],
    1,
    False,
    plot_lims=[-5, 5, -5, 5])

# Logger.
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logger = Logger(os.path.join(LOG_DIRECTORY, 'intersection_example.pkl'))

# Set up ILQSolver.
solver = ILQSolver(dynamics,
                   [car1_cost, car2_cost, car3_cost],
                   stacked_x0,
                   [car1_Ps, car2_Ps, car3_Ps],
                   [car1_alphas, car2_alphas, car3_alphas],
                   0.02,
                   0.1,
                   None,
                   visualizer,
                   None)

start = timer()
solver.run()
dt = timer() - start
print("Run took %f seconds" % dt)
