################################################################################
#
# Test script for timing simple LQ games.
#
################################################################################

import numpy as np
from timeit import default_timer as timer
from solve_lq_game import solve_lq_game

DT = 0.1
HORIZON = 10.0
NUM_TIMESTEPS = int(HORIZON / DT)

# 2D point mass.
A = np.eye(2) + DT * np.array([
    [0.0, 1.0],
    [0.0, 0.0]
])

B1 = DT * np.array([
    [0.05],
    [1.0]
])

B2 = DT * np.array([
    [0.032],
    [0.11]
])

As = [A] * NUM_TIMESTEPS
B1s = [B1] * NUM_TIMESTEPS
B2s = [B2] * NUM_TIMESTEPS

# State costs.
Q1 = np.diag([1.0, 1.0]); Q1s = [Q1] * NUM_TIMESTEPS
Q2 = -Q1; Q2s = [Q2] * NUM_TIMESTEPS
l1 = np.zeros((2, 1)); l1s = [l1] * NUM_TIMESTEPS
l2 = -l1; l2s = [l2] * NUM_TIMESTEPS

# Control costs.
R11 = np.eye(1); R11s = [R11] * NUM_TIMESTEPS
R12 = np.zeros((1, 1)); R12s = [R12] * NUM_TIMESTEPS
R21 = np.zeros((1, 1)); R21s = [R21] * NUM_TIMESTEPS
R22 = np.eye(1); R22s = [R22] * NUM_TIMESTEPS

# timing results
time_sum = 0
n_samples = 1000
print("Running experiments with %d samples:" % (n_samples))
for i in range(n_samples):
    t_start = timer()
    [P1s, P2s], [alpha1s, alpha2s] = solve_lq_game(
        As, [B1s, B2s],
        [Q1s, Q2s], [l1s, l2s], [[R11s, R12s], [R21s, R22s]])
    t_end = timer()
    time_sum += t_end - t_start

print("Average runtime over %d samples was: %f milli seconds." % (n_samples,
      time_sum / n_samples * 1e3))
