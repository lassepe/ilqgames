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
// Semiquadratic cost on distance from a polyline.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/cost/time_invariant_cost.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <tuple>

namespace ilqgames {

float SemiquadraticPolyline2Cost::Evaluate(const VectorXf& input) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  // Compute signed squared distance by finding closest point.
  float signed_squared_distance;
  polyline_.ClosestPoint(Point2(input(xidx_), input(yidx_)),
                         &signed_squared_distance);

  // Check which side we're on.
  if (!IsActive(signed_squared_distance)) return 0.0;

  // Handle orientation.
  const float signed_distance = sgn(signed_squared_distance) *
                                std::sqrt(std::abs(signed_squared_distance));
  const float diff = signed_distance - threshold_;
  return 0.5 * weight_ * diff * diff;
}

void SemiquadraticPolyline2Cost::Quadraticize(const VectorXf& input,
                                              MatrixXf* hess,
                                              VectorXf* grad) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  CHECK_NOTNULL(hess);
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());

  // Compute signed squared distance by finding closest point.
  float signed_squared_distance;
  const Point2 closest_point = polyline_.ClosestPoint(
      Point2(input(xidx_), input(yidx_)), &signed_squared_distance);

  /// Check if cost is active.
  if (!IsActive(signed_squared_distance)) return;

  // Compute signed distance and diff.
  const float signed_distance = sgn(signed_squared_distance) *
                                std::sqrt(std::abs(signed_squared_distance));
  const float diff = signed_distance - threshold_;

  // Handle Hessian first.
  (*hess)(xidx_, xidx_) += weight_;
  (*hess)(yidx_, yidx_) += weight_;

  // Maybe handle gradient.
  if (grad) {
    CHECK_EQ(input.size(), grad->size());

    // Unpack current position and find closest point.
    const Point2 current_position(input(xidx_), input(yidx_));
    Point2 relative = current_position - closest_point;
    relative *= std::abs(diff) * relative.norm();

    (*grad)(xidx_) += weight_ * relative.x();
    (*grad)(yidx_) += weight_ * relative.y();
  }
}

}  // namespace ilqgames