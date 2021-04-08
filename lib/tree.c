#include "tree.h"
#include "algo.h"
#include "utils.h"

#include <stdio.h>

#ifndef PROFILE
void tree_print(tree_t const *tree_nodes, ssize_t size, ssize_t n_tree_nodes,
                ssize_t n_points) {
    fprintf(stdout, "%zd %zd\n", N_DIMENSIONS, n_tree_nodes + n_points);
    for (ssize_t i = 0; i < size; ++i) {
        tree_t const *t = tree_index_to_ptr(tree_nodes, i);
        if (t->t_radius == 0) {
            continue;
        }
        /*
        LOG("printing %zd {\n"
            "   type:   %d,\n"
            "   radius: %lf,\n"
            "   left:   %p,\n"
            "   right:  %p,\n"
            "}", i, t->t_type, t->t_radius, t->t_left, t->t_right);
            */

        if (tree_has_left_leaf(t) != 0) {
            double const *left = (double const *)t->t_left;
            fprintf(stdout, "%zd -1 -1 %.6lf", tree_left_node_idx(i), 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(stdout, " %lf", left[j]);
            }
            fputc('\n', stdout);
        }

        if (tree_has_right_leaf(t) != 0) {
            double const *right = (double const *)t->t_right;
            fprintf(stdout, "%zd -1 -1 %.6lf", tree_right_node_idx(i), 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(stdout, " %lf", right[j]);
            }
            fputc('\n', stdout);
        }

        fprintf(stdout, "%zd %zd %zd %.6lf", i, tree_left_node_idx(i),
                tree_right_node_idx(i), t->t_radius);
        for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
            fprintf(stdout, " %lf", t->t_center[j]);
        }
        fputc('\n', stdout);
    }
}
#endif

static ssize_t tree_build_aux(tree_t *tree_nodes, double const **points,
                              ssize_t idx, ssize_t l, ssize_t r,
                              strategy_t find_points) {
    assert(r - l > 1, "1-sized trees are out of scope");

    tree_t *t = tree_index_to_ptr(tree_nodes, idx);
    // LOG("building tree node %zd: %p [%zd, %zd[ -> L = %zd, R = %zd", idx,
    //(void*)t, l, r, tree_left_node_idx(idx), tree_right_node_idx(idx));

    divide_point_set(points, l, r, find_points, t->t_center);
    t->t_radius = compute_radius(points, l, r, t->t_center);

    ssize_t m = (l + r) / 2;
    if (r - l == 2) {
        t->t_type = TREE_TYPE_BOTH_LEAF;
        t->t_left = (void *)points[l];
        t->t_right = (void *)points[r - 1];
        return 1;
    }

    if (r - l == 3) {
        t->t_type = TREE_TYPE_LEFT_LEAF;
        t->t_left = (void *)points[l];
        t->t_right =
            (void *)tree_index_to_ptr(tree_nodes, tree_right_node_idx(idx));

        return 1 + tree_build_aux(tree_nodes, points, tree_right_node_idx(idx),
                                  m, r, find_points);
    }

    t->t_type = TREE_TYPE_INNER;
    t->t_left = (void *)tree_index_to_ptr(tree_nodes, tree_left_node_idx(idx));
    t->t_right =
        (void *)tree_index_to_ptr(tree_nodes, tree_right_node_idx(idx));

    ssize_t l_children = tree_build_aux(
        tree_nodes, points, tree_left_node_idx(idx), l, m, find_points);
    ssize_t r_children = tree_build_aux(
        tree_nodes, points, tree_right_node_idx(idx), m, r, find_points);

    return 1 + l_children + r_children;
}

// returns the number of inner nodes (ie: tree_t structs)
//
ssize_t tree_build(tree_t *tree_nodes, double const **points, ssize_t n_points,
                   strategy_t find_points) {
    return tree_build_aux(tree_nodes, points, 0, 0, n_points, find_points);
}
