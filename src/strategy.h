/***
 * strategies for dividing a set of points.
 *
 * a `strategy` ends up being a way of choosing two points to define a line
 * this is sufficient to produce variations in the ball algorithm
 */

#pragma once
#include <stdlib.h>
#include "geometry.h"
#include "types.h"
#include "utils.h"

// a stratey receives a list of points, a range [l, r[, and out-ptrs for the
// result. it returns the distance between these two points
//
typedef double (*strategy_t)(double const **, ssize_t, ssize_t, ssize_t *,
                             ssize_t *);

extern ssize_t N_DIMENSIONS;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

// Exhaustively find the two most distant points
// time = n^2
//
static double most_distant(double const **points, ssize_t l, ssize_t r,
                           ssize_t *a, ssize_t *b) {
    double max_dist = 0;
    for (ssize_t i = l; i < r - 1; ++i) {
        for (ssize_t j = i + 1; j < r; ++j) {
            double dist = distance_squared(points[i], points[j]);
            if (dist > max_dist) {
                *a = i;
                *b = j;
                max_dist = dist;
            }
        }
    }

    return max_dist;
}

// Find the two most distant points approximately
// a is the furthest point from points[l]
// b is the furthes point from a
//
// time = 2*n
//
static double most_distant_approx(double const **points, ssize_t l, ssize_t r,
                                  ssize_t *a, ssize_t *b) {
    double dist_l_a = 0;
    for (ssize_t i = l + 1; i < r; ++i) {
        double dist = distance_squared(points[l], points[i]);
        if (dist > dist_l_a) {
            dist_l_a = dist;
            *a = i;
        }
    }

    double dist_a_b = 0;
    for (ssize_t i = l; i < r; ++i) {
        if (i == *a) {
            continue;
        }
        double dist = distance_squared(points[*a], points[i]);
        if (dist > dist_a_b) {
            dist_a_b = dist;
            *b = i;
        }
    }

    return dist_a_b;
}

#ifdef WORST_PARALLEL
static double most_distant_approx_parallel(double const **points, ssize_t l, ssize_t r,
                                  ssize_t *a, ssize_t *b, ssize_t available) {
    double dist_l_a = 0;
    double dist_a_b = 0;

    ssize_t max_i_a = 0;
    double max_dist_a = 0.0;

    ssize_t max_i_b = 0;
    double max_dist_b = 0.0;

#pragma omp parallel firstprivate(max_dist_a, max_i_a, max_dist_b, max_i_b) shared(a, dist_l_a, dist_a_b, l, r) num_threads(available)
    {
    //fprintf(stderr, "%zd %zd %d %d\n", l, r, omp_get_team_num(), omp_get_thread_num());
    #pragma omp for nowait
        for (ssize_t i = l + 1; i < r; ++i) {
            double dist = distance_squared(points[l], points[i]);
            if (dist > max_dist_a) {
                max_dist_a = dist;
                max_i_a = i;
            }
        }
    #pragma omp critical
        if (max_dist_a > dist_l_a) {
            dist_l_a = max_dist_a;
            *a = max_i_a;
        }

    #pragma omp barrier

    #pragma omp for nowait
        for (ssize_t i = l; i < r; ++i) {
            if (i == *a) {
                continue;
            }
            double dist = distance_squared(points[*a], points[i]);
            if (dist > max_dist_b) {
                max_dist_b = dist;
                max_i_b = i;
            }
        }

    #pragma omp critical
        if (max_dist_b > dist_a_b) {
            dist_a_b = max_dist_b;
            *b = max_i_b;
        }
    }

    return dist_a_b;
}
#endif

// Find the two most distant points approximately using the centroid methon
// a is the furthest point from the centroid (arith mean of points[l..r])
// b is the furthes point from a
//
// time = 3*n
//
static double most_distant_centroid(double const **points, ssize_t l, ssize_t r,
                                    ssize_t *a, ssize_t *b) {
    double *centroid = xcalloc((size_t)N_DIMENSIONS, sizeof(double));

    for (ssize_t d = 0; d < N_DIMENSIONS; d++) {
        for (ssize_t i = l; i < r; i++) {
            centroid[d] += points[i][d];
        }
        centroid[d] /= (double)N_DIMENSIONS;
    }

    double dist_l_a = 0;
    for (ssize_t i = l + 1; i < r; ++i) {
        double dist = distance_squared(centroid, points[i]);
        if (dist > dist_l_a) {
            dist_l_a = dist;
            *a = i;
        }
    }

    double dist_a_b = 0;
    for (ssize_t i = l; i < r; ++i) {
        if (i == *a) {
            continue;
        }
        double dist = distance_squared(points[*a], points[i]);
        if (dist > dist_a_b) {
            dist_a_b = dist;
            *b = i;
        }
    }

    free(centroid);

    return dist_a_b;
}

// Select random points
//
static double select_random(double const **points, ssize_t l, ssize_t r,
                            ssize_t *a, ssize_t *b) {
    *a = l + (rand() % (r - l));
    // to ensure that *b != *a, we increment *a with a random value in [1,n-1],
    // where n is r - l.
    ssize_t increment = 1 + rand() % (r - l - 1);
    // we do the increment with wrap-around
    *b = l + ((*a - l) + increment) % (r - l);
    return distance_squared(points[*a], points[*b]);
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
