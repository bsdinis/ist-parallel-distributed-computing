/***
 * strategies for dividing a set of points.
 *
 * a `strategy` ends up being a way of choosing two points to define a line
 * this is sufficient to produce variations in the ball algorithm
 */

#pragma once
#include "types.h"

// a stratey receives a list of points, a range [l, r[, and out-ptrs for the result.
// it returns the distance between these two points
//
typedef double (*strategy_t)(double const **, ssize_t, ssize_t, ssize_t *, ssize_t *);

// Exhaustively find the two most distant points
// time = n^2
//
double most_distant(double const ** points, ssize_t l, ssize_t r, ssize_t *a, ssize_t *b);

// Find the two most distant points approximately
// a is the furthest point from points[l]
// b is the furthes point from a
//
// time = 2*n
//
double most_distant_approx(double const ** points, ssize_t l, ssize_t r, ssize_t *a, ssize_t *b);

// Find the two most distant points approximately using the centroid methon
// a is the furthest point from the centroid (arith mean of points[l..r])
// b is the furthes point from a
//
// time = 3*n
//
double most_distant_centroid(double const ** points, ssize_t l, ssize_t r, ssize_t *a, ssize_t *b);

// Select random points
//
double select_random(double const ** points, ssize_t l, ssize_t r, ssize_t *a, ssize_t *b);
