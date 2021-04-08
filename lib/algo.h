/**
 * Functions for the algorithm
 */

#pragma once
#include "types.h"
#include "strategy.h"

// Divide a point set, finding its center (for the ball algorithm)
// will reorder the points in the set.
//
// Receives a strategy, which decides how to find two of the points to define the line
//
void divide_point_set(double const **points, ssize_t l, ssize_t r, strategy_t find_points, double *center);

// Compute radius of a ball, given its center
//
double compute_radius(double const **points, ssize_t l, ssize_t r,
                             double const *center);
