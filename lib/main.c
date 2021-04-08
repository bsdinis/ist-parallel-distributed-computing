#include "main.h"

#include <errno.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tree.h"
#include "utils.h"

ssize_t N_DIMENSIONS = 0;

#ifndef RANGE
#define RANGE 10
#endif  // RANGE

// parse the arguments
static double const **parse_args(int argc, char *argv[], ssize_t *n_points) {
    if (argc != 4) {
        KILL("usage: %s <n_dimensions> <n_points> <seed>", argv[0]);
    }

    errno = 0;
    N_DIMENSIONS = strtoll(argv[1], NULL, 0);
    if (errno != 0) {
        KILL("failed to parse %s as integer: %s", argv[1], strerror(errno));
    }
    if (N_DIMENSIONS < 2) {
        KILL("Illegal number of dimensions (%zd), must be above 1.",
             N_DIMENSIONS);
    }

    errno = 0;
    *n_points = strtoll(argv[2], NULL, 0);
    if (errno != 0) {
        KILL("failed to parse %s as integer: %s", argv[2], strerror(errno));
    }
    if (*n_points < 1) {
        KILL("Illegal number of points (%zd), must be above 0.", *n_points);
    }

    errno = 0;
    uint32_t seed = (uint32_t)strtoul(argv[3], NULL, 0);
    if (errno != 0) {
        KILL("failed to parse %s as integer: %s", argv[3], strerror(errno));
    }

    srand(seed);

    // Lemma: the number of inner nodes, m, will be Theta(n), worst case.
    // Proof.
    // 1. m < n: there are more leaves than inner nodes. This is true by
    // induction, stemming from the fact that there are always 2 childs per
    // inner node.
    //
    // 2. if n == 2^k: m = 2^k - 1.
    // By induction.
    // Base: a tree with 2^0 nodes has 0 inner nodes.
    // Induction Step: consider two trees, each with 2^k nodes.
    //                 by assumption, they have 2^k - 1 inner nodes.
    //                 we join them using a root, creating a tree
    //                 with 2^{k+1} leaves and 2^{k + 1} - 2 + 1 inner nodes
    //                 (note the addition of the root).
    //
    double *pt_arr =
        xmalloc(sizeof(double) * (size_t)N_DIMENSIONS * (size_t)*n_points);
    double **pt_ptr = xmalloc(sizeof(double *) * (size_t)*n_points);

    // double ** pt_points = xmalloc(sizeof(double*) * (size_t)*n_points );

    for (ssize_t i = 0; i < *n_points; i++) {
        for (ssize_t j = 0; j < N_DIMENSIONS; j++) {
            pt_arr[i * (N_DIMENSIONS) + j] =
                RANGE * ((double)rand()) / RAND_MAX;
        }
        pt_ptr[i] = &pt_arr[i * (N_DIMENSIONS)];
    }

    return (double const **)pt_ptr;
}

int strategy_main(int argc, char **argv, strategy_t strategy) {
    double const begin = omp_get_wtime();

    ssize_t n_points = 0;
    double const **points = parse_args(argc, argv, &n_points);
    double const *point_values = points[0];

    // As discussed in parse_args, the number of inner nodes is
    // at most the number of leaves of the tree.
    //
    tree_t *tree_nodes =
        xcalloc((size_t)(2 * n_points), tree_sizeof());  // FMA initialization

    ssize_t n_tree_nodes = tree_build(tree_nodes, points, n_points, strategy);

    fprintf(stderr, "%.1lf\n", omp_get_wtime() - begin);

#ifndef PROFILE
    tree_print(tree_nodes, 2 * n_points, n_tree_nodes, n_points);
#endif

    free((void *)point_values);
    free(points);
    free(tree_nodes);

    return 0;
}
