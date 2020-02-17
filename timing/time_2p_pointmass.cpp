#include <ilqgames/cost/player_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/solver/solve_lq_game.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>
#include <math.h>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using nanoseconds = std::chrono::nanoseconds;

using namespace ilqgames;

namespace {

// Utility class for time-invariant linear system.
class TwoPlayerPointMass1D : public MultiPlayerDynamicalSystem {
 public:
  ~TwoPlayerPointMass1D() {}
  TwoPlayerPointMass1D(Time time_step)
      : MultiPlayerDynamicalSystem(2, time_step), A_(2, 2), B1_(2), B2_(2) {
    A_ = MatrixXf::Zero(2, 2);
    A_(0, 1) = 1.0;

    B1_(0) = 0.05;
    B1_(1) = 1.0;

    B2_(0) = 0.032;
    B2_(1) = 0.11;
  }

  // Getters.
  Dimension UDim(PlayerIndex player_index) const { return 1; }
  PlayerIndex NumPlayers() const { return 2; }

  // Time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x,
                    const std::vector<VectorXf>& us) const {
    return A_ * x + B1_ * us[0] + B2_ * us[1];
  }

  // Discrete-time Jacobian linearization.
  LinearDynamicsApproximation Linearize(Time t, const VectorXf& x,
                                        const std::vector<VectorXf>& us) const {
    LinearDynamicsApproximation linearization(*this);

    linearization.A += A_ * time_step_;
    linearization.Bs[0] = B1_ * time_step_;
    linearization.Bs[1] = B2_ * time_step_;
    return linearization;
  }

 private:
  // Continuous-time dynamics.
  MatrixXf A_;
  VectorXf B1_, B2_;
};  // class TwoPlayerPointMass1D

}  // anonymous namespace

// Reset with nonzero nominal state and control.
std::vector<PlayerCost> constructPlayerCost(float nominal = 0.0) {
  auto pc = std::vector<PlayerCost>();
  constexpr float kRelativeCostScaling = 0.1;

  PlayerCost player1_cost;
  player1_cost.AddStateCost(
      std::make_shared<QuadraticCost>(1.0, -1, nominal));
  player1_cost.AddControlCost(
      0, std::make_shared<QuadraticCost>(1.0, -1, nominal));
  player1_cost.AddControlCost(
      1, std::make_shared<QuadraticCost>(kRelativeCostScaling, -1, nominal));
  pc.push_back(player1_cost);

  PlayerCost player2_cost;
  player2_cost.AddStateCost(
      std::make_shared<QuadraticCost>(kRelativeCostScaling, -1, nominal));
  player2_cost.AddControlCost(
      0, std::make_shared<QuadraticCost>(kRelativeCostScaling, -1, nominal));
  player2_cost.AddControlCost(
      1, std::make_shared<QuadraticCost>(1.0, -1, nominal));
  pc.push_back(player2_cost);

  return pc;
}


int main(int argc, char *argv[])
{
  // timing parameters
  static constexpr size_t n_samples = 1000;
  std::chrono::duration<long, std::ratio<1, 1000000000>> time_sum;

  // Time parameters.
  static constexpr Time kTimeStep = 0.1;
  static constexpr Time kTimeHorizon = 10.0;
  static constexpr size_t kNumTimeSteps = static_cast<size_t>(kTimeHorizon / kTimeStep);

  // Dynamics.
  std::unique_ptr<TwoPlayerPointMass1D> dynamics_;
  // Operating point.
  std::unique_ptr<OperatingPoint> operating_point_;
  // Player costs.
  const auto player_costs_ = constructPlayerCost();
  // Linearization and quadraticization.
  LinearDynamicsApproximation linearization_;
  std::vector<QuadraticCostApproximation> quadraticizations_;
  // Solution to LQ game.
  std::vector<Strategy> lq_solution_;

  /* ----------------------------------- setup -----------------------------------*/
  dynamics_.reset(new TwoPlayerPointMass1D(kTimeStep));
  // Set linearization and quadraticizations.
  linearization_ = dynamics_->Linearize(
      0.0, VectorXf::Zero(2), {VectorXf::Zero(1), VectorXf::Zero(1)});
  // Set a zero operating point.
  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics_->NumPlayers(), 0.0));
  for (size_t kk = 0; kk < kNumTimeSteps; kk++) {
    operating_point_->xs[kk] =
        MatrixXf::Zero(dynamics_->XDim(), dynamics_->XDim());
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
      operating_point_->us[kk][ii] = VectorXf::Zero(dynamics_->UDim(ii));
  }
  // quadraticization
  quadraticizations_.clear();
  quadraticizations_.push_back(player_costs_[0].Quadraticize(
      0.0, operating_point_->xs[0], operating_point_->us[0]));
  quadraticizations_.push_back(player_costs_[1].Quadraticize(
      0.0, operating_point_->xs[0], operating_point_->us[0]));

  /*------------------------------- time the solver ------------------------------*/

  for (int i = 0; i < n_samples; i++)
  {
    Clock::time_point t_start = Clock::now();
    lq_solution_ = SolveLQGame(
        *dynamics_,
        std::vector<LinearDynamicsApproximation>(kNumTimeSteps, linearization_),
        std::vector<std::vector<QuadraticCostApproximation>>(
          kNumTimeSteps, quadraticizations_));
    Clock::time_point t_end = Clock::now();
    const auto dt = t_end - t_start;
    std::cout << 1e-6*std::chrono::duration_cast<nanoseconds>(dt).count() << std::endl;
    time_sum += dt;
  }
  std::cout << "Average over " << n_samples << " was: "
    << 1e-6*std::chrono::duration_cast<nanoseconds>(time_sum).count()/n_samples
    << " milliseconds." << std::endl;
  return 0;
}
