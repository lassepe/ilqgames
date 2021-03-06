/*
 * Copyright (c) 2019, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Base class specifying the problem interface for managing calls to the core
// ILQGame solver. Specific examples will be derived from this class.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace ilqgames {

std::shared_ptr<SolverLog> Problem::Solve(Time max_runtime) {
  CHECK_NOTNULL(solver_.get());
  CHECK_NOTNULL(strategies_.get());
  CHECK_NOTNULL(operating_point_.get());

  // Create empty log.
  std::shared_ptr<SolverLog> log = CreateNewLog();

  // Solver the problem.
  OperatingPoint final_operating_point(*operating_point_);
  std::vector<Strategy> final_strategies(*strategies_);
  if (!solver_->Solve(x0_, *operating_point_, *strategies_,
                      &final_operating_point, &final_strategies, log.get(),
                      max_runtime)) {
    LOG(WARNING) << "Solver failed.";
    return log;
  }

  // Store these new strategies/operating point.
  strategies_->swap(final_strategies);
  operating_point_->swap(final_operating_point);

  LOG(INFO) << "Solver succeeded.";
  return log;
}

void Problem::SetUpNextRecedingHorizon(const VectorXf& x0, Time t0,
                                       Time planner_runtime) {
  CHECK_NOTNULL(strategies_.get());
  CHECK_NOTNULL(operating_point_.get());
  CHECK_GE(planner_runtime, 0.0);
  CHECK_LE(planner_runtime + t0, operating_point_->t0 + solver_->TimeHorizon());
  CHECK_GE(t0, operating_point_->t0);

  const MultiPlayerIntegrableSystem& dynamics = solver_->Dynamics();

  // Integrate x0 forward from t0 by approximately planner_runtime to get
  // actual initial state. Integrate up to the next discrete timestep, then
  // integrate for an integer number of discrete timesteps until at least
  // 'planner_runtime' has elapsed (done by rounding).
  const Time relative_t0 = t0 - operating_point_->t0;
  const size_t first_integration_timestep =
      static_cast<size_t>(relative_t0 / solver_->TimeStep());
  const Time remaining_time_this_step =
      solver_->TimeStep() * (first_integration_timestep + 1) - relative_t0;
  const size_t num_steps_to_integrate =
      static_cast<size_t>(0.5 +
                          std::max(planner_runtime - remaining_time_this_step,
                                   static_cast<Time>(0.0)) /
                              solver_->TimeStep());
  const size_t last_integration_timestep =
      first_integration_timestep + num_steps_to_integrate;

  VectorXf x =
      dynamics.IntegrateToNextTimeStep(t0, x0, *operating_point_, *strategies_);
  x = dynamics.Integrate(first_integration_timestep + 1,
                         last_integration_timestep, x, *operating_point_,
                         *strategies_);

  // Find index of nearest state in the existing plan to this state.
  const auto nearest_iter =
      std::min_element(operating_point_->xs.begin(), operating_point_->xs.end(),
                       [&dynamics, &x](const VectorXf& x1, const VectorXf& x2) {
                         return dynamics.DistanceBetween(x, x1) <
                                dynamics.DistanceBetween(x, x2);
                       });

  // Set initial state to this state.
  // NOTE: we are currently *not* interpolating when finding the closest point.
  x0_ = dynamics.Stitch(*nearest_iter, x);

  // Set initial time to first timestamp in new problem.
  const size_t first_timestep_in_new_problem =
      std::distance(operating_point_->xs.begin(), nearest_iter);
  operating_point_->t0 =
      t0 + remaining_time_this_step +
      solver_->TimeStep() * num_steps_to_integrate;  // planner_runtime;

  // Update all costs to have the correct initial time.
  Cost::ResetInitialTime(operating_point_->t0);

  // NOTE: when we call 'solve' on this new operating point it will
  // automatically end up starting at 'x0', so there is no need to enforce that
  // here.

  // Populate strategies and opeating point for the remainder of the
  // existing plan, reusing the old operating point when possible.
  for (size_t kk = first_timestep_in_new_problem;
       kk < operating_point_->xs.size(); kk++) {
    const size_t kk_new_problem = kk - first_timestep_in_new_problem;

    // Set current state and controls in operating point.
    operating_point_->xs[kk_new_problem] = operating_point_->xs[kk];
    operating_point_->us[kk_new_problem].swap(operating_point_->us[kk]);

    // Set current stategy.
    for (auto& strategy : *strategies_) {
      strategy.Ps[kk_new_problem] = strategy.Ps[kk];
      strategy.alphas[kk_new_problem] = strategy.alphas[kk];
    }
  }

  // Set new operating point controls and strategies to zero and propagate state
  // forward accordingly.
  for (size_t kk = operating_point_->xs.size() - first_timestep_in_new_problem;
       kk < operating_point_->xs.size(); kk++) {
    for (size_t ii = 0; ii < dynamics.NumPlayers(); ii++) {
      (*strategies_)[ii].Ps[kk].setZero();
      (*strategies_)[ii].alphas[kk].setZero();
      operating_point_->us[kk][ii].setZero();
    }

    operating_point_->xs[kk] = dynamics.Integrate(
        operating_point_->t0 + solver_->ComputeTimeStamp(kk - 1),
        solver_->TimeStep(), operating_point_->xs[kk - 1],
        operating_point_->us[kk - 1]);
  }
}

void Problem::OverwriteSolution(const OperatingPoint& operating_point,
                                const std::vector<Strategy>& strategies) {
  CHECK_NOTNULL(operating_point_.get());
  CHECK_NOTNULL(strategies_.get());

  x0_ = operating_point.xs[0];
  operating_point_->t0 = operating_point.t0;

  const size_t num_copies =
      std::min(operating_point.xs.size(), operating_point_->xs.size());
  for (size_t kk = 0; kk < num_copies; kk++) {
    operating_point_->xs[kk] = operating_point.xs[kk];
    operating_point_->us[kk] = operating_point.us[kk];

    for (PlayerIndex ii = 0; ii < strategies_->size(); ii++) {
      (*strategies_)[ii].Ps[kk] = strategies[ii].Ps[kk];
      (*strategies_)[ii].alphas[kk] = strategies[ii].alphas[kk];
    }
  }
}

std::shared_ptr<SolverLog> Problem::CreateNewLog() const {
  return std::make_shared<SolverLog>(solver_->TimeStep());
}

}  // namespace ilqgames
