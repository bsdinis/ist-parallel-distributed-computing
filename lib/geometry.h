/***
 * generic geometric functions
 */

#pragma once

// computes the square of the distance between to points
//
double distance_squared(double const *pt_1, double const *pt_2);

// computes the inner product between the difference of two points and a vector
// this avoids actually constructing the difference between the two points
//
double diff_inner_product(double const *pt, double const *a,
                                 double const *b_minus_a);

// a . b
double inner_product(double const *a, double const *b);

