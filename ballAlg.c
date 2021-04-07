// ballAlg
// seraial implementation
//

#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char *argv[]) {
    double const begin = omp_get_wtime();

    ssize_t n_points;
    double const **points = parse_args(argc, argv, &n_points);
    double const *point_values = points[0];

    // As discussed in gen_tree_points, the number of inner nodes is
    // assymptotically the number of leaves of the tree
    // However, since our id assignment is sparse, we double that size, because
    // even the sparse assignment is bounded by 2*n_points
    //
    tree_t *tree_nodes =
        xcalloc(2 * (size_t)n_points, tree_sizeof());  // FMA initialization

#ifndef PROFILE
    ssize_t n_tree_nodes = tree_build(tree_nodes, points, 0, 0, n_points);
#else
    tree_build(tree_nodes, points, 0, 0, n_points)
#endif

    fprintf(stderr, "%.1lf\n", omp_get_wtime() - begin);

#ifndef PROFILE
    tree_print(tree_nodes, max_idx, n_tree_nodes, n_points);
#endif

    free((void *)point_values);
    free(points);
    free(tree_nodes);
}