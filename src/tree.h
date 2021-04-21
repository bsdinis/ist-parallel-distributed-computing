#pragma once

#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "algo.h"
#include "strategy.h"
#include "types.h"
#include "utils.h"

extern ssize_t N_DIMENSIONS;

typedef enum {
    TREE_TYPE_INNER = 0,       // 0b00: both left and right are inners
    TREE_TYPE_LEFT_LEAF = 1,   // 0b01: left is a leaf
    TREE_TYPE_RIGHT_LEAF = 2,  // 0b10: right is a leaf // WILL NEVER OCCOUR
    TREE_TYPE_BOTH_LEAF = 3,   // 0b11: both are leaves
} tree_type_t;

typedef struct {
    tree_type_t t_type;
    double t_radius;
    ssize_t t_left;   // idx to tree_t or double
    ssize_t t_right;  // idx to tree_t or double
    double t_center[];
} tree_t;

// Accessors for the type of the node in the tree
//
static inline bool tree_is_inner(tree_t const *t) { return t->t_type == 0; }
static inline bool tree_has_left_leaf(tree_t const *t) {
    return (((uint32_t)t->t_type) & (uint32_t)1) != 0;
}
static inline bool tree_has_right_leaf(tree_t const *t) {
    return (((uint32_t)t->t_type) & (uint32_t)2) != 0;
}

// functions for assigning nodes
//
static inline ssize_t tree_left_node_idx(ssize_t parent) {
    return 2 * parent + 1;
}
static inline ssize_t tree_right_node_idx(ssize_t parent) {
    return 2 * parent + 2;
}

// Calling sizeof(tree_t) is always wrong, because tree nodes have an FMA
// (flexible member array) This function makes that correct
//
static inline size_t tree_sizeof() {
    return sizeof(tree_t) + sizeof(double) * (size_t)N_DIMENSIONS;
}

// Safety: this function needs to return a pointer aligned to 8 bytes
//
// This is always true, because the original pointer (tree_vec) is also aligned
// to an 8 byte boundary, and since tree_sizeof() a multiple of 8, the result
// will also be.
//
// However, clang has no visibility of this given the multiple casts
// (gcc does)
//
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

static inline tree_t *tree_index_to_ptr(tree_t const *tree_vec, ssize_t idx) {
    return (tree_t *)(((uint8_t *)tree_vec) + (size_t)idx * tree_sizeof());
}

#pragma clang diagnostic pop

// Safety: this assumes both pointers are to the same contiguous array
//         and that base_ptr < ptr
static inline ssize_t tree_ptr_to_index(tree_t const *base_ptr,
                                        tree_t const *ptr) {
    return (ptr == NULL)
               ? -1
               : (ssize_t)((size_t)(ptr - base_ptr) / (tree_sizeof()));
}

#ifndef PROFILE
static void tree_print(tree_t const *tree_nodes, ssize_t tree_size,
                       double const **points, ssize_t n_points) {
    for (ssize_t i = 0; i < tree_size; ++i) {
        tree_t const *t = tree_index_to_ptr(tree_nodes, i);
        if (t->t_radius == 0) {
            continue;
        }

        n_points++;
    }

    fprintf(stdout, "%zd %zd\n", N_DIMENSIONS, n_points);
    for (ssize_t i = 0; i < tree_size; ++i) {
        tree_t const *t = tree_index_to_ptr(tree_nodes, i);
        if (t->t_radius == 0) {
            continue;
        }
        /*
        LOG("printing %zd {\n"
            "   type:   %d,\n"
            "   radius: %lf,\n"
            "   left:   %zd,\n"
            "   right:  %zd,\n"
            "}", i, t->t_type, t->t_radius, t->t_left, t->t_right);
            */

        if (tree_has_left_leaf(t) != 0) {
            fprintf(stdout, "%zd -1 -1 %.6lf", tree_left_node_idx(i), 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(stdout, " %lf", points[t->t_left][j]);
            }
            fputc('\n', stdout);
        }

        if (tree_has_right_leaf(t) != 0) {
            fprintf(stdout, "%zd -1 -1 %.6lf", tree_right_node_idx(i), 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(stdout, " %lf", points[t->t_right][j]);
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

static void tree_build_aux(tree_t *tree_nodes, double const **points,
                           ssize_t idx, ssize_t l, ssize_t r,
                           strategy_t find_points, ssize_t ava, ssize_t depth) {
    assert(r - l > 1, "1-sized trees are out of scope");

    tree_t *t = tree_index_to_ptr(tree_nodes, idx);
    // LOG("building tree node %zd: %p [%zd, %zd[ -> L = %zd, R = %zd", idx,
    //(void*)t, l, r, tree_left_node_idx(idx), tree_right_node_idx(idx));

    // double const begin = omp_get_wtime();
    divide_point_set(points, l, r, find_points, t->t_center, ava + 1);
    t->t_radius = compute_radius(points, l, r, t->t_center);
    // fprintf(stderr, "%zd %.12lf\n", depth, omp_get_wtime() - begin);

    ssize_t m = (l + r) / 2;
    if (r - l == 2) {
        t->t_type = TREE_TYPE_BOTH_LEAF;
        t->t_left = l;
        t->t_right = r - 1;
        return;
    }

    if (r - l == 3) {
        t->t_type = TREE_TYPE_LEFT_LEAF;
        t->t_left = l;
        t->t_right = tree_right_node_idx(idx);

        tree_build_aux(tree_nodes, points, t->t_right, m, r, find_points, ava,
                       depth + 1);
        return;
    }

    t->t_type = TREE_TYPE_INNER;
    t->t_left = tree_left_node_idx(idx);
    t->t_right = tree_right_node_idx(idx);

    if (ava > 0) {  // Parallel
#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            {
                /*
                 fprintf(stderr, "level: %d | team: %d | id %d | ava: %zd\n",
                 omp_get_active_level(), omp_get_team_num(),
                 omp_get_thread_num(), ava - (1 << depth));
                 */
                tree_build_aux(tree_nodes, points, t->t_left, l, m, find_points,
                               ava - (1 << (size_t)depth), depth + 1);
            }

#pragma omp section
            {
                /*
                 fprintf(stderr, "level: %d | team: %d | id %d | ava: %zd\n",
                 omp_get_active_level(), omp_get_team_num(),
                 omp_get_thread_num(), ava - (1 << (size_t)depth));
                 */
                tree_build_aux(tree_nodes, points, t->t_right, m, r,
                               find_points, ava - (1 << depth), depth + 1);
            }
        }
    } else {  // Serial
        tree_build_aux(tree_nodes, points, t->t_left, l, m, find_points, 0,
                       depth + 1);
        tree_build_aux(tree_nodes, points, t->t_right, m, r, find_points, 0,
                       depth + 1);
    }
}

// Compute the tree
//
static void tree_build(tree_t *tree_nodes, double const **points,
                       ssize_t n_points, strategy_t find_points) {
    omp_set_max_active_levels(omp_get_max_threads());
    tree_build_aux(tree_nodes, points, 0 /* idx */, 0 /* l */, n_points /* r */,
                   find_points /* strategy */,
                   omp_get_max_threads() - 1 /* available threads */,
                   0 /* depth */);
}
