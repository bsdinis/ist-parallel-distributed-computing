/**
 * Functions for the algorithm
 */

#pragma once
#include "geometry.h"
#include "strategy.h"
#include "types.h"
#include "utils.h"

#include <math.h>
#include <stdlib.h>

extern ssize_t N_DIMENSIONS;

static inline void swap_ptr(void **a, void **b);

// ----------------------------------------------------------
// Aux functions
// ----------------------------------------------------------

static inline void swap_double(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

// Finds the max double in a vector
//
static double find_max(double *vec, size_t size) {
    double max = 0.0;
    for (size_t i = 0; i < size; i++) {
        if (vec[i] > max) {
            max = vec[i];
        }
    }
    return max;
}

// Partitions the vector
// using best of three pivot
//
static size_t partition(double *vec, size_t l, size_t r) {
    double *lo = vec;
    double *hi = &vec[r - 1];
    double *mid = &vec[(l + r) / 2];

    // Picks pivout from 3 numbers
    // leaves the 3 numbers ordered
    if (*mid < *lo) {
        swap_double(mid, lo);
    }
    if (*hi < *mid) {
        swap_double(mid, hi);
        if (*mid < *lo) {
            swap_double(mid, lo);
        }
    }

    if (r - l <= 3) {  // already ordered
        return (size_t)(mid - vec);
    }

    double pivout = *mid;
    swap_double(mid, hi - 1);  // store pivout away

    size_t i = l + 1;

    for (size_t j = l + 1; j < r - 2; j++) {  // -2 (pivout and hi)
        if (vec[j] <= pivout) {
            double temp1 = vec[i];
            double temp2 = vec[j];
            vec[i] = temp2;
            vec[j] = temp1;
            i++;
        }
    }
    double temp1 = vec[i];
    double temp2 = vec[r - 2];
    vec[i] = temp2;
    vec[r - 2] = temp1;

    return i;
}

// QuickSelect algorithm
// Finds the kth_smallest index in array
//
static double qselect(double *vec, size_t l, size_t r, size_t k) {
    // find the partition

    size_t p = partition(vec, l, r);

    if (p == k || r - l <= 3) return vec[k];

    if (p > k) return qselect(vec, l, p, k);

    return qselect(vec, p + 1, r, k);
}

// Find the median value of a vector
//
static double find_median(double *vec, ssize_t size) {
    size_t k = (size_t)size / 2;
    double median = qselect(vec, 0, (size_t)size, k);
    if (size % 2 == 0) {
        median = (median + find_max(vec, k)) / 2;
    }
    return median;
}

// Swap two pointers
// Note there are two temporary variables. This is by design.
// Using one (or no) variables may seem fashionable, but it introduces data
// dependencies. And, as we all know, the CPU pipeline is not a fan of data
// dependencies
//
static inline void swap_ptr(void **a, void **b) {
    void *tmp1 = *a;
    void *tmp2 = *b;
    *a = tmp2;
    *b = tmp1;
}

// Partition a set of points based on the median value (in the products array).
// This reorders the points in the [l, r[ range.
//
static void partition_on_median(double const **points, ssize_t l, ssize_t r,
                                double const *products, double median) {
    ssize_t i = 0;
    ssize_t j = r - l - 1;
    ssize_t k = (r - l) / 2;
    while (i < j) {
        while (i < j && products[i] < median) {
            i++;
        }
        while (i < j && products[j] > median) {
            j--;
        }
        if (i < j) {
            if (products[i] == median) {  // i and j will swap
                k = j;
            } else if (products[j] == median) {
                k = i;
            }
            swap_ptr((void **)&points[l + i], (void **)&points[l + j]);
            i++;
            j--;
        }
    }
    ssize_t m = (r + l) / 2;
    swap_ptr((void **)&points[l + k],
             (void **)&points[m]);  // ensure medium is on the right set
}

// ----------------------------------------------------------
// Algorithm
// ----------------------------------------------------------

// Divide a point set, finding its center (for the ball algorithm)
// will reorder the points in the set.
//
static void divide_point_set(double const **points, ssize_t l, ssize_t r,
                             strategy_t find_points, double *center,
                             ssize_t available) {
    ssize_t a = l;
    ssize_t b = l;

    // 2 * n
    // No point in parallelizing: requires too much synchronization overhead
    double dist = find_points(points, l, r, &a, &b);

    // points[a] may change after the partition
    double const *a_ptr = points[a];
    double *b_minus_a = xmalloc((size_t)N_DIMENSIONS * sizeof(double));
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        b_minus_a[i] = points[b][i] - points[a][i];
    }

    double *products = xmalloc((size_t)(r - l) * 2 * sizeof(double));
    double *products_aux = products + r - l;

    // n
    if (false && available > 1) {
#pragma omp parallel for num_threads(available)
        for (ssize_t i = 0; i < r - l; ++i) {
            products[i] =
                diff_inner_product(points[l + i], points[a], b_minus_a);
            products_aux[i] = products[i];
        }
    } else {
        for (ssize_t i = 0; i < r - l; ++i) {
            products[i] =
                diff_inner_product(points[l + i], points[a], b_minus_a);
            products_aux[i] = products[i];
        }
    }

    // O(n)
    // No point in parallelizing: requires too much synchronization overhead
    double median = find_median(products, (r - l));

    // O(n)
    // Not possible to parallelize
    partition_on_median(points, l, r, products_aux, median);

    double normalized_median = median / dist;
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        center[i] =
            a_ptr[i] + (b_minus_a[i] *       // NOLINT: this does not see that
                        normalized_median);  // we fully initialize b_minus_a
    }

    free(b_minus_a);
    free(products);
}

// Compute radius of a ball, given its center
// As this requires synchronization to compute in parallel, we have observed
// slowdowns from trying to parallelize this.
//
// Returns radius
//
static double compute_radius(double const **points, ssize_t l, ssize_t r,
                             double const *center) {
    double max_dist_sq = 0.0;
    for (ssize_t i = l; i < r; i++) {
        double dist = distance_squared(center, points[i]);
        if (dist > max_dist_sq) {
            max_dist_sq = dist;
        }
    }

    return sqrt(max_dist_sq);
}
