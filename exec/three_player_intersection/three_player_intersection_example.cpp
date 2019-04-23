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
// Three player intersection example. Ordering is given by the following:
// (P1, P2, P3) = (Car 1, Car 2, Pedestrian).
//
///////////////////////////////////////////////////////////////////////////////

#include "three_player_intersection_example.h"
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/dynamics/single_player_unicycle_4d.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilqgame.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/log.h>
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

// Car inter-axle distance.
static constexpr float kInterAxleLength = 5.0;  // m

// Cost weights.
static constexpr float kLaneCostWeight = 100.0;

// Initial state.
static constexpr float kP1InitialX = -5.0;   // m
static constexpr float kP2InitialX = -10.0;  // m
static constexpr float kP3InitialX = -12.0;  // m

static constexpr float kP1InitialY = -30.0;  // m
static constexpr float kP2InitialY = 50.0;   // m
static constexpr float kP3InitialY = 15.0;   // m

static constexpr float kP1InitialHeading = M_PI_2;   // rad
static constexpr float kP2InitialHeading = -M_PI_2;  // rad
static constexpr float kP3InitialHeading = 0.0;      // rad

static constexpr float kP1InitialSpeed = 1.0;  // m/s
static constexpr float kP2InitialSpeed = 1.0;  // m/s
static constexpr float kP3InitialSpeed = 1.0;  // m/s

// State dimensions.
static constexpr Dimension kP1XIdx = 0;
static constexpr Dimension kP1YIdx = 1;
static constexpr Dimension kP1HeadingIdx = 2;
static constexpr Dimension kP1VIdx = 4;
static constexpr Dimension kP2XIdx = 5;
static constexpr Dimension kP2YIdx = 6;
static constexpr Dimension kP2HeadingIdx = 7;
static constexpr Dimension kP2VIdx = 9;
static constexpr Dimension kP3XIdx = 10;
static constexpr Dimension kP3YIdx = 11;
static constexpr Dimension kP3HeadingIdx = 12;
static constexpr Dimension kP3VIdx = 13;
}  // anonymous namespace

ThreePlayerIntersectionExample::ThreePlayerIntersectionExample()
    : x_idxs_({kP1XIdx, kP2XIdx, kP3XIdx}),
      y_idxs_({kP1YIdx, kP2YIdx, kP3YIdx}),
      heading_idxs_({kP1HeadingIdx, kP2HeadingIdx, kP3HeadingIdx}) {
  // Create dynamics.
  const std::shared_ptr<ConcatenatedDynamicalSystem> dynamics(
      new ConcatenatedDynamicalSystem(
          {std::make_shared<SinglePlayerCar5D>(kInterAxleLength),
           std::make_shared<SinglePlayerCar5D>(kInterAxleLength),
           std::make_shared<SinglePlayerUnicycle4D>()}));

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
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), dynamics));

  // Set up costs for all players.
  const Polyline2 lane1(
      {Point2(kP1InitialX, -100.0), Point2(kP1InitialX, 100.0)});
  const Polyline2 lane2({Point2(kP2InitialX, -100.0), Point2(kP2InitialX, 18.0),
                         Point2(kP2InitialX + 0.5, 15.0),
                         Point2(kP2InitialX + 1.0, 14.0),
                         Point2(kP2InitialX + 3.0, 12.5),
                         Point2(kP2InitialX + 6.0, 12.0), Point2(100.0, 12.0)});
  const Polyline2 lane3(
      {Point2(-100.0, kP3InitialY), Point2(100.0, kP3InitialY)});

  const QuadraticPolyline2Cost p1_lane_cost(kLaneCostWeight, lane1,
                                            {kP1XIdx, kP1YIdx});

  // Set up solver.
  solver_.reset(new ILQGame(dynamics, player_costs, kTimeHorizon, kTimeStep));
}

}  // namespace ilqgames
