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

// Compare two doubles in a qsort(3) compatible way
//
static inline int cmp_double(void const *a, void const *b) {
    double a_double = *(double const *)a;
    double b_double = *(double const *)b;
    if (a_double < b_double) {
        return -1;
    }
    if (a_double > b_double) {
        return 1;
    }
    return 0;
}

// Finds the max double in a vector
//
static double find_max(double* vec, size_t size) {
    double max = 0.0;
    for (size_t i = 0; i < size; i++) {
        if (vec[i] > max) {
            max = vec[i];
        }
    }
    return max;
}

// Partitions the vector
//
static size_t partition_non(double* vec, size_t l, size_t r)
{
    size_t i = l;
    double pivout = vec[r - 1]; // TODO not random
    for (size_t j = l; j < r; j++) {
        if (vec[j] <= pivout) {
            double temp1 = vec[i];
            double temp2 = vec[j];
            vec[i] = temp2;
            vec[j] = temp1;
            i++;
        }
    }
    double temp1 = vec[i];
    double temp2 = vec[r - 1];
    vec[i] = temp2;
    vec[r-1] = temp1;
    return i;
}

// QuickSelect algorithm
// Finds the kth_smallest index in array
//
static double kth_smallest(double *vec, size_t l, size_t r, size_t k) {
    // find the partition
    size_t partition = partition_non(vec, l, r);

    if (partition == k)
        return vec[partition];

    if (partition > k)
        return kth_smallest(vec, l, partition - 1, k);

    return kth_smallest(vec, partition + 1, r, k);
}

// Find the median value of a vector
//
// static double find_median(double *vec, ssize_t size) {
//     qsort(vec, (size_t)size, sizeof(double), cmp_double);
//     return (size % 2 == 0) ? (vec[(size - 2) / 2] + vec[size / 2]) / 2
//                            : vec[(size - 1) / 2];
// }
static double find_median(double *vec, ssize_t size) {
    size_t k = (size_t)size/2;
    double median = kth_smallest(vec, 0, (size_t)size, k);
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
    ssize_t m = (r - l) / 2;
    swap_ptr((void **)&points[l + k],
             (void **)&points[l + m]);  // ensure medium is on the right set
}

// ----------------------------------------------------------
// Algorithm
// ----------------------------------------------------------

// Divide a point set, finding its center (for the ball algorithm)
// will reorder the points in the set.
//
static void divide_point_set(double const **points, ssize_t l, ssize_t r,
                             strategy_t find_points, double *center) {
    ssize_t a = l;
    ssize_t b = l;
    double dist = find_points(points, l, r, &a, &b);

    double const *a_ptr =
        points[a];  // points[a] may change after the partition
    double *b_minus_a = xmalloc((size_t)N_DIMENSIONS * sizeof(double));
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        b_minus_a[i] = points[b][i] - points[a][i];
    }

    double *products = xmalloc((size_t)(r - l) * 2 * sizeof(double));
    double *products_aux = products + r - l;
    for (ssize_t i = 0; i < r - l; ++i) {
        products[i] = diff_inner_product(points[l + i], points[a], b_minus_a);
        products_aux[i] = products[i];
    }

    // O(n)
    double median = find_median(products, (r - l));

    // O(n)
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
