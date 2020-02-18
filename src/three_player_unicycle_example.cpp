///////////////////////////////////////////////////////////////////////////////
//
// Three player unicycle navigation example for timing benchmark
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/cost/weighted_convex_proximity_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/dynamics/single_player_car_6d.h>
#include <ilqgames/dynamics/single_player_unicycle_4d.h>
#include <ilqgames/examples/three_player_unicycle_example.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
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

static constexpr float kOmegaCostWeight = 10.0;
static constexpr float kACostWeight = 10.0;

static constexpr float kMaxVCostWeight = 50.0;
static constexpr float kNominalVCostWeight = 30.0;
static constexpr float kGoalCostWeight = 300.0;

static constexpr float kMinProximity = 1.2;
static constexpr float kP1ProximityCostWeight = 50.0;
static constexpr float kP2ProximityCostWeight = 50.0;
static constexpr float kP3ProximityCostWeight = 50.0;
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
static constexpr float kP1MaxV = 2.0;  // m/s
static constexpr float kP2MaxV = 2.0;  // m/s
static constexpr float kP3MaxV = 2.0;   // m/s
static constexpr float kMinV = -0.05;     // m/s

static constexpr float kP1NominalV = 0;  // m/s
static constexpr float kP2NominalV = 0;  // m/s
static constexpr float kP3NominalV = 0;  // m/s

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
using P1 = SinglePlayerUnicycle4D;
using P2 = SinglePlayerUnicycle4D;
using P3 = SinglePlayerUnicycle4D;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1HeadingIdx = P1::kThetaIdx;
static const Dimension kP1VIdx = P1::kVIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2HeadingIdx = P1::kNumXDims + P2::kThetaIdx;
static const Dimension kP2VIdx = P1::kNumXDims + P2::kVIdx;

static const Dimension kP3XIdx = P1::kNumXDims + P2::kNumXDims + P3::kPxIdx;
static const Dimension kP3YIdx = P1::kNumXDims + P2::kNumXDims + P3::kPyIdx;
static const Dimension kP3HeadingIdx = P1::kNumXDims + P2::kNumXDims + P3::kThetaIdx;
static const Dimension kP3VIdx = P1::kNumXDims + P2::kNumXDims + P3::kVIdx;

// Control dimensions.
static const Dimension kP1OmegaIdx = 0;
static const Dimension kP1AIdx = 1;
static const Dimension kP2OmegaIdx = 0;
static const Dimension kP2AIdx = 1;
static const Dimension kP3OmegaIdx = 0;
static const Dimension kP3AIdx = 1;
}  // anonymous namespace

ThreePlayerUnicycleExample::ThreePlayerUnicycleExample(
    const SolverParams& params) {
  // Create dynamics.
  const std::shared_ptr<const ConcatenatedDynamicalSystem> dynamics(
      new ConcatenatedDynamicalSystem(
          {std::make_shared<P1>(),
           std::make_shared<P2>(),
           std::make_shared<P3>()},
          kTimeStep));

  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics->XDim());
  x0_(kP1XIdx) = kP1InitialX;
  x0_(kP1YIdx) = kP1InitialY;
  x0_(kP1HeadingIdx) = kP1InitialHeading;
  x0_(kP1VIdx) = kP1InitialSpeed;
  x0_(kP2XIdx) = kP2InitialX;
  x0_(kP2YIdx) = kP2InitialY;
  x0_(kP2HeadingIdx) = kP2InitialHeading;
  x0_(kP2VIdx) = kP2InitialSpeed;
  x0_(kP3XIdx) = kP3InitialX;
  x0_(kP3YIdx) = kP3InitialY;
  x0_(kP3HeadingIdx) = kP3InitialHeading;
  x0_(kP3VIdx) = kP3InitialSpeed;

  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics->XDim(),
                              dynamics->UDim(ii));

  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), 0.0, dynamics));

  // Set up costs for all players.
  PlayerCost p1_cost(kStateRegularization, kControlRegularization);
  PlayerCost p2_cost(kStateRegularization, kControlRegularization);
  PlayerCost p3_cost(kStateRegularization, kControlRegularization);

  // Max/min/nominal speed costs.
  const auto p1_min_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP1VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p1_max_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP1VIdx, kP1MaxV, kOrientedRight, "MaxV");
  const auto p1_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP1VIdx, kP1NominalV, "NominalV");
  p1_cost.AddStateCost(p1_min_v_cost);
  p1_cost.AddStateCost(p1_max_v_cost);
  p1_cost.AddStateCost(p1_nominal_v_cost);

  const auto p2_min_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP2VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p2_max_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP2VIdx, kP2MaxV, kOrientedRight, "MaxV");
  const auto p2_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP2VIdx, kP2NominalV, "NominalV");
  p2_cost.AddStateCost(p2_min_v_cost);
  p2_cost.AddStateCost(p2_max_v_cost);
  p2_cost.AddStateCost(p2_nominal_v_cost);

  const auto p3_min_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP3VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p3_max_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP3VIdx, kP3MaxV, kOrientedRight, "MaxV");
  const auto p3_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP3VIdx, kP3NominalV, "NominalV");
  p3_cost.AddStateCost(p3_min_v_cost);
  p3_cost.AddStateCost(p3_max_v_cost);
  p3_cost.AddStateCost(p3_nominal_v_cost);

  // Penalize control effort.
  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP1OmegaIdx, 0.0, "Steering");
  const auto p1_a_cost =
      std::make_shared<QuadraticCost>(kACostWeight, kP1AIdx, 0.0, "Acceleration");
  p1_cost.AddControlCost(0, p1_omega_cost);
  p1_cost.AddControlCost(0, p1_a_cost);

  const auto p2_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP2OmegaIdx, 0.0, "Steering");
  const auto p2_a_cost =
      std::make_shared<QuadraticCost>(kACostWeight, kP2AIdx, 0.0, "Acceleration");
  p2_cost.AddControlCost(1, p2_omega_cost);
  p2_cost.AddControlCost(1, p2_a_cost);

  const auto p3_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP3OmegaIdx, 0.0, "Steering");
  const auto p3_a_cost = std::make_shared<QuadraticCost>(kACostWeight, kP3AIdx,
                                                         0.0, "Acceleration");
  p3_cost.AddControlCost(2, p3_omega_cost);
  p3_cost.AddControlCost(2, p3_a_cost);

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
      new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  const std::shared_ptr<ProxCost> p1p3_proximity_cost(
      new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  p1_cost.AddStateCost(p1p2_proximity_cost);
  p1_cost.AddStateCost(p1p3_proximity_cost);

  const std::shared_ptr<ProxCost> p2p1_proximity_cost(
      new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p2p3_proximity_cost(
      new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  p2_cost.AddStateCost(p2p1_proximity_cost);
  p2_cost.AddStateCost(p2p3_proximity_cost);

  const std::shared_ptr<ProxCost> p3p1_proximity_cost(
      new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p3p2_proximity_cost(
      new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  p3_cost.AddStateCost(p3p1_proximity_cost);
  p3_cost.AddStateCost(p3p2_proximity_cost);

  // Set up solver.
  solver_.reset(new ILQSolver(dynamics, {p1_cost, p2_cost, p3_cost},
                              kTimeHorizon, params));
}

inline std::vector<float> ThreePlayerUnicycleExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx), x(kP3XIdx)};
}

inline std::vector<float> ThreePlayerUnicycleExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx), x(kP3YIdx)};
}

inline std::vector<float> ThreePlayerUnicycleExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1HeadingIdx), x(kP2HeadingIdx), x(kP3HeadingIdx)};
}

}  // namespace ilqgames
