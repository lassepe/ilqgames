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
// Three player *flat* intersection example. Ordering is given by the following:
// (P1, P2, P3) = (Car 1, Car 2, Pedestrian).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/dynamics/concatenated_flat_system.h>
#include <ilqgames/dynamics/single_player_flat_unicycle_4d.h>
#include <ilqgames/examples/three_player_flat_unicycle_example.h>
#include <ilqgames/solver/ilq_flat_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/initialize_along_route.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <math.h>
#include <memory>
#include <vector>

namespace ilqgames {

namespace {

// Time.
static constexpr Time kTimeStep = 0.1;      // s
static constexpr Time kTimeHorizon = 10.0;  // s
static constexpr size_t kNumTimeSteps =
    static_cast<size_t>(kTimeHorizon / kTimeStep);

// Cost weights.
static constexpr float kStateRegularization = 5.0;
static constexpr float kControlRegularization = 5.0;

// input cost
static constexpr float kUnicycleAuxCostWeight = 10.0;
static constexpr float kNominalVCostWeight = 50.0;
static constexpr float kMaxVCostWeight = 50.0;
static constexpr float kProximityCostWeight = 50.0;
static constexpr float kMinProximity = 1.2;
static constexpr float kGoalCostWeight = 300;

using ProxCost = ProximityCost;

static constexpr bool kOrientedRight = true;

// Goal points.
static constexpr float kP1GoalX = 3.0;   // m
static constexpr float kP1GoalY = 0.0;  // m

static constexpr float kP2GoalX = -0.1;  // m
static constexpr float kP2GoalY = -3.0;   // m

static constexpr float kP3GoalX = 0.1;  // m
static constexpr float kP3GoalY = 3.0;   // m

// Nominal and max speed.
static constexpr float kMinV = -2.0;     // m/s
static constexpr float kMaxV = 2.0;     // m/s
static constexpr float kNominalV = 0.0;

// Initial state.
static constexpr float kP1InitialX = -3.0;   // m
static constexpr float kP1InitialY = 0.0;  // m
static constexpr float kP1InitialHeading = 0;   // rad
static constexpr float kP1InitialSpeed = 0.05;   // m/s

static constexpr float kP2InitialX = -0.1;  // m
static constexpr float kP2InitialY = 3.0;   // m
static constexpr float kP2InitialHeading = -M_PI_2;  // rad
static constexpr float kP2InitialSpeed = 0.1;   // m/s

static constexpr float kP3InitialX = 0.1;  // m
static constexpr float kP3InitialY = -3.0;   // m
static constexpr float kP3InitialHeading = M_PI_2;      // rad
static constexpr float kP3InitialSpeed = 0.1;  // m/s

// State dimensions.
using P1 = SinglePlayerFlatUnicycle4D;
using P2 = SinglePlayerFlatUnicycle4D;
using P3 = SinglePlayerFlatUnicycle4D;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1HeadingIdx = P1::kThetaIdx;
static const Dimension kP1VIdx = P1::kVIdx;
static const Dimension kP1VxIdx = P1::kVxIdx;
static const Dimension kP1VyIdx = P1::kVyIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2HeadingIdx = P1::kNumXDims + P2::kThetaIdx;
static const Dimension kP2VIdx = P1::kNumXDims + P2::kVIdx;
static const Dimension kP2VxIdx = P1::kNumXDims + P2::kVxIdx;
static const Dimension kP2VyIdx = P1::kNumXDims + P2::kVyIdx;

static const Dimension kP3XIdx = P1::kNumXDims + P2::kNumXDims + P3::kPxIdx;
static const Dimension kP3YIdx = P1::kNumXDims + P2::kNumXDims + P3::kPyIdx;
static const Dimension kP3HeadingIdx = P1::kNumXDims + P2::kNumXDims + P3::kThetaIdx;
static const Dimension kP3VIdx = P1::kNumXDims + P2::kNumXDims + P3::kVIdx;
static const Dimension kP3VxIdx = P1::kNumXDims + P2::kNumXDims + P3::kVxIdx;
static const Dimension kP3VyIdx = P1::kNumXDims + P2::kNumXDims + P3::kVyIdx;

// Control dimensions.
static const Dimension kP1OmegaIdx = 0;
static const Dimension kP1AIdx = 1;
static const Dimension kP2OmegaIdx = 0;
static const Dimension kP2AIdx = 1;
static const Dimension kP3OmegaIdx = 0;
static const Dimension kP3AIdx = 1;
}  // anonymous namespace

ThreePlayerFlatUnicycleExample::ThreePlayerFlatUnicycleExample(
    const SolverParams& params) {
  // Create dynamics.
  dynamics_.reset(new ConcatenatedFlatSystem(
      {std::make_shared<P1>(), std::make_shared<P2>(), std::make_shared<P3>()},
      kTimeStep));

  // Set up initial state.
  VectorXf x0 = VectorXf::Zero(dynamics_->XDim());
  x0(kP1XIdx) = kP1InitialX;
  x0(kP1YIdx) = kP1InitialY;
  x0(kP1HeadingIdx) = kP1InitialHeading;
  x0(kP1VIdx) = kP1InitialSpeed;
  x0(kP2XIdx) = kP2InitialX;
  x0(kP2YIdx) = kP2InitialY;
  x0(kP2HeadingIdx) = kP2InitialHeading;
  x0(kP2VIdx) = kP2InitialSpeed;
  x0(kP3XIdx) = kP3InitialX;
  x0(kP3YIdx) = kP3InitialY;
  x0(kP3HeadingIdx) = kP3InitialHeading;
  x0(kP3VIdx) = kP3InitialSpeed;

  x0_ = dynamics_->ToLinearSystemState(x0);

  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics_->XDim(),
                              dynamics_->UDim(ii));

  operating_point_.reset(new OperatingPoint(
      kNumTimeSteps, dynamics_->NumPlayers(), 0.0, dynamics_));

  // Set up costs for all players.
  PlayerCost p1_cost(kStateRegularization, kControlRegularization);
  PlayerCost p2_cost(kStateRegularization, kControlRegularization);
  PlayerCost p3_cost(kStateRegularization, kControlRegularization);

  // Max/min/nominal speed costs.
  const std::shared_ptr<SemiquadraticCost> p1_min_vx_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP1VxIdx, kMinV, !kOrientedRight,
        "MinVx"));
  const std::shared_ptr<SemiquadraticCost> p1_max_vx_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP1VxIdx, kMaxV, kOrientedRight,
        "MaxVx"));

  const std::shared_ptr<SemiquadraticCost> p1_min_vy_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP1VyIdx, kMinV, !kOrientedRight,
        "MinVy"));
  const std::shared_ptr<SemiquadraticCost> p1_max_vy_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP1VyIdx, kMaxV, kOrientedRight,
        "MaxVy"));

  const std::shared_ptr<QuadraticCost> p1_nominal_vx_cost(
      new QuadraticCost(kNominalVCostWeight, kP1VxIdx, kNominalV, "NominalVx"));
  const std::shared_ptr<QuadraticCost> p1_nominal_vy_cost(
      new QuadraticCost(kNominalVCostWeight, kP1VyIdx, kNominalV, "NominalVy"));

  p1_cost.AddStateCost(p1_min_vx_cost);
  p1_cost.AddStateCost(p1_max_vx_cost);
  p1_cost.AddStateCost(p1_min_vy_cost);
  p1_cost.AddStateCost(p1_max_vy_cost);
  p1_cost.AddStateCost(p1_nominal_vx_cost);
  p1_cost.AddStateCost(p1_nominal_vy_cost);



  // Max/min/nominal speed costs.
  const std::shared_ptr<SemiquadraticCost> p2_min_vx_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP2VxIdx, kMinV, !kOrientedRight,
        "MinVx"));
  const std::shared_ptr<SemiquadraticCost> p2_max_vx_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP2VxIdx, kMaxV, kOrientedRight,
        "MaxVx"));

  const std::shared_ptr<SemiquadraticCost> p2_min_vy_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP2VyIdx, kMinV, !kOrientedRight,
        "MinVy"));
  const std::shared_ptr<SemiquadraticCost> p2_max_vy_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP2VyIdx, kMaxV, kOrientedRight,
        "MaxVy"));

  const std::shared_ptr<QuadraticCost> p2_nominal_vx_cost(
      new QuadraticCost(kNominalVCostWeight, kP2VxIdx, kNominalV, "NominalVx"));
  const std::shared_ptr<QuadraticCost> p2_nominal_vy_cost(
      new QuadraticCost(kNominalVCostWeight, kP2VyIdx, kNominalV, "NominalVy"));

  p2_cost.AddStateCost(p2_min_vx_cost);
  p2_cost.AddStateCost(p2_max_vx_cost);
  p2_cost.AddStateCost(p2_min_vy_cost);
  p2_cost.AddStateCost(p2_max_vy_cost);
  p2_cost.AddStateCost(p2_nominal_vx_cost);
  p2_cost.AddStateCost(p2_nominal_vy_cost);


  // Max/min/nominal speed costs.
  const std::shared_ptr<SemiquadraticCost> p3_min_vx_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP3VxIdx, kMinV, !kOrientedRight,
        "MinVx"));
  const std::shared_ptr<SemiquadraticCost> p3_max_vx_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP3VxIdx, kMaxV, kOrientedRight,
        "MaxVx"));

  const std::shared_ptr<SemiquadraticCost> p3_min_vy_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP3VyIdx, kMinV, !kOrientedRight,
        "MinVy"));
  const std::shared_ptr<SemiquadraticCost> p3_max_vy_cost(
      new SemiquadraticCost(kMaxVCostWeight, kP3VyIdx, kMaxV, kOrientedRight,
        "MaxVy"));

  const std::shared_ptr<QuadraticCost> p3_nominal_vx_cost(
      new QuadraticCost(kNominalVCostWeight, kP3VxIdx, kNominalV, "NominalVx"));
  const std::shared_ptr<QuadraticCost> p3_nominal_vy_cost(
      new QuadraticCost(kNominalVCostWeight, kP3VyIdx, kNominalV, "NominalVy"));

  p3_cost.AddStateCost(p3_min_vx_cost);
  p3_cost.AddStateCost(p3_max_vx_cost);
  p3_cost.AddStateCost(p3_min_vy_cost);
  p3_cost.AddStateCost(p3_max_vy_cost);
  p3_cost.AddStateCost(p3_nominal_vx_cost);
  p3_cost.AddStateCost(p3_nominal_vy_cost);



  // Penalize control effort.
  constexpr Dimension kApplyInAllDimensions = -1;
  const auto unicycle_aux_cost = std::make_shared<QuadraticCost>(
      kUnicycleAuxCostWeight, kApplyInAllDimensions, 0.0, "Auxiliary Input");
  p1_cost.AddControlCost(0, unicycle_aux_cost);
  p2_cost.AddControlCost(1, unicycle_aux_cost);
  p3_cost.AddControlCost(2, unicycle_aux_cost);

  // Goal costs.
  constexpr float kFinalTimeWindow = 0.15;  // s
  const auto p1_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP1XIdx, kP1GoalX),
      kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p1_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP1YIdx, kP1GoalY),
      kTimeHorizon - kFinalTimeWindow, "GoalY");
  p1_cost.AddStateCost(p1_goalx_cost);
  p1_cost.AddStateCost(p1_goaly_cost);

  const auto p2_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP2XIdx, kP2GoalX),
      kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p2_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP2YIdx, kP2GoalY),
      kTimeHorizon - kFinalTimeWindow, "GoalY");
  p2_cost.AddStateCost(p2_goalx_cost);
  p2_cost.AddStateCost(p2_goaly_cost);

  const auto p3_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP3XIdx, kP3GoalX),
      kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p3_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP3YIdx, kP3GoalY),
      kTimeHorizon - kFinalTimeWindow, "GoalY");
  p3_cost.AddStateCost(p3_goalx_cost);
  p3_cost.AddStateCost(p3_goaly_cost);

  // Pairwise proximity costs.
  const std::shared_ptr<ProxCost> p1p2_proximity_cost(
      new ProxCost(kProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  const std::shared_ptr<ProxCost> p1p3_proximity_cost(
      new ProxCost(kProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  p1_cost.AddStateCost(p1p2_proximity_cost);
  p1_cost.AddStateCost(p1p3_proximity_cost);

  const std::shared_ptr<ProxCost> p2p1_proximity_cost(
      new ProxCost(kProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p2p3_proximity_cost(
      new ProxCost(kProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  p2_cost.AddStateCost(p2p1_proximity_cost);
  p2_cost.AddStateCost(p2p3_proximity_cost);

  const std::shared_ptr<ProxCost> p3p1_proximity_cost(
      new ProxCost(kProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p3p2_proximity_cost(
      new ProxCost(kProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  p3_cost.AddStateCost(p3p1_proximity_cost);
  p3_cost.AddStateCost(p3p2_proximity_cost);

  // Set up solver.
  solver_.reset(new ILQFlatSolver(dynamics_, {p1_cost, p2_cost, p3_cost},
                                  kTimeHorizon, params));
}

inline std::vector<float> ThreePlayerFlatUnicycleExample::Xs(
    const VectorXf& xi) const {
  return {xi(kP1XIdx), xi(kP2XIdx), xi(kP3XIdx)};
}

inline std::vector<float> ThreePlayerFlatUnicycleExample::Ys(
    const VectorXf& xi) const {
  return {xi(kP1YIdx), xi(kP2YIdx), xi(kP3YIdx)};
}

inline std::vector<float> ThreePlayerFlatUnicycleExample::Thetas(
    const VectorXf& xi) const {
  const VectorXf x = dynamics_->FromLinearSystemState(xi);
  return {x(kP1HeadingIdx), x(kP2HeadingIdx), x(kP3HeadingIdx)};
}

}  // namespace ilqgames
