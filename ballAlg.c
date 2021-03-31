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

#ifndef RANGE
#define RANGE 10
#endif

#define INSERTION_SORT_CONSTANT (192)
#define PIVOT_POOL_N (5)

#ifndef ssize_t
#define ssize_t __ssize_t
#endif

#define KILL(...)                                                \
    {                                                            \
        fprintf(stderr, "[ERROR] %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
        exit(EXIT_FAILURE);                                      \
        __builtin_unreachable();                                 \
    }

#define WARN(...)
#define LOG(...)
#define DBG(stmt)
#ifndef NDEBUG
#undef DBG
#define DBG(stmt) stmt

#undef WARN
#define WARN(...)                                                \
    {                                                            \
        fprintf(stderr, "[WARN]  %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
    }

#undef LOG
#define LOG(...)                                                 \
    {                                                            \
        fprintf(stderr, "[LOG]   %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
    }

#endif /* NDEBUG */

/* util functions for memory management
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
/* checked malloc
 */
static void *_priv_xmalloc(char const *file, int lineno, size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL && size != 0) {
        fprintf(stderr, "[ERROR] %s:%d | xmalloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return ptr;
}

/* checked realloc
 */
static void *_priv_xrealloc(char const *file, int lineno, void *ptr,
                            size_t size) {
    void *new_ptr = realloc(ptr, size);
    if (size != 0 && new_ptr == NULL && ptr != NULL) {
        fprintf(stderr, "[ERROR] %s:%d | xrealloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return new_ptr;
}

// checked calloc
void *_priv_xcalloc(char const *file, int lineno, size_t nmemb, size_t size) {
    void *ptr = calloc(nmemb, size);
    if (ptr == NULL && size != 0) {
        fprintf(stderr, "[ERROR] %s:%d | xcalloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return ptr;
}

#define xmalloc(size) _priv_xmalloc(__FILE__, __LINE__, size)
#define xrealloc(ptr, size) _priv_xrealloc(__FILE__, __LINE__, ptr, size)
#define xcalloc(nmemb, size) _priv_xcalloc(__FILE__, __LINE__, nmemb, size)

#pragma clang diagnostic pop

/* we define a better assert than that from the stdilb
 */
#ifndef NDEBUG
#define assert(pred, msg)                                                 \
    if (!(pred)) {                                                        \
        fprintf(stderr, "[ASSERT] %s:%d | %s | %s\n", __FILE__, __LINE__, \
                #pred, msg);                                              \
        exit(EXIT_FAILURE);                                               \
    }
#else
#define assert(pred, msg)        \
    if (!(pred)) {               \
        __builtin_unreachable(); \
    }
#endif /* NDEBUG */

static ssize_t N_DIMENSIONS = 0;
DBG(static ssize_t N_POINTS = 0;)

typedef enum {
    TREE_TYPE_INNER = 0,       // 0b00: both left and right are inners
    TREE_TYPE_LEFT_LEAF = 1,   // 0b01: left is a leaf
    TREE_TYPE_RIGHT_LEAF = 2,  // 0b10: right is a leaf // WILL NEVER OCCOUR
    TREE_TYPE_BOTH_LEAF = 3,   // 0b11: both are leaves
} tree_type_t;

typedef struct {
    tree_type_t t_type;
    double t_radius;
    void *t_left;   // tree_t or double
    void *t_right;  // tree_t or double
    double t_center[];
} tree_t;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
static inline bool tree_is_inner(tree_t const *t) { return t->t_type == 0; }
#pragma clang diagnostic pop
static inline bool tree_has_left_leaf(tree_t const *t) {
    return (((uint32_t)t->t_type) & (uint32_t)1) != 0;
}
static inline bool tree_has_right_leaf(tree_t const *t) {
    return (((uint32_t)t->t_type) & (uint32_t)2) != 0;
}

// consistent functions for assignind indices
static inline ssize_t tree_left_node_idx(ssize_t parent) {
    return 2 * parent + 1;
}
static inline ssize_t tree_right_node_idx(ssize_t parent) {
    return 2 * parent + 2;
}

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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
// Safety: this assumes both pointers are to the same contiguous array
//         and that base_ptr < ptr
static inline ssize_t tree_ptr_to_index(tree_t const *base_ptr,
                                        tree_t const *ptr) {
    return (ptr == NULL) ? -1
                         : (ssize_t)((size_t)((uint8_t const *)ptr -
                                              (uint8_t const *)base_ptr) /
                                     (tree_sizeof()));
}
#pragma clang diagnostic pop

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

    srandom(seed);

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
                RANGE * ((double)random()) / RAND_MAX;
        }
        pt_ptr[i] = &pt_arr[i * (N_DIMENSIONS)];
    }

    return (double const **)pt_ptr;
}

// D**2 = Sum {v=1 to N_DIMENSIONS} (pt1_x - pt2_x)**2
static inline double distance_squared(double const *pt_1, double const *pt_2) {
    double d_s = 0;
    for (ssize_t i = 0; i < N_DIMENSIONS; i++) {
        double aux = (pt_1[i] - pt_2[i]);
        d_s += aux * aux;
    }
    return d_s;
}

// Partition a set of points, finding its center
// Assumes the points are ordered by the first coordinate
// l and r define an interval [l, r[
//
static void divide_point_set(double const **points, ssize_t l, ssize_t r,
                             double *center) {
    memset(center, 0, (size_t)N_DIMENSIONS * sizeof(double));
    if ((r - l) % 2 == 0) {
        center[0] = (points[(r + l - 1) / 2][0] + points[(r + l) / 2][0]) / 2;
    } else {
        center[0] = points[(r + l - 1) / 2][0];
    }
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

static ssize_t tree_build_aux(tree_t *tree_nodes, double const **points,
                              ssize_t *idx_ptr, ssize_t l, ssize_t r,
                              ssize_t *max_idx) {
    assert(r - l > 1, "1-sized trees are out of scope");
    ssize_t idx = *max_idx;
    *idx_ptr = idx;
    (*max_idx)++;
    DBG(assert(idx < N_POINTS,
               "there can never be more inner nodes than leaves");)

    tree_t *t = tree_index_to_ptr(tree_nodes, idx);
    // LOG("building tree node %zd: %p [%zd, %zd[ -> L = %zd, R = %zd", idx,
    //(void*)t, l, r, tree_left_node_idx(idx), tree_right_node_idx(idx));

    divide_point_set(points, l, r, t->t_center);
    t->t_radius = compute_radius(points, l, r, t->t_center);

    ssize_t m = (l + r) / 2;
    if (r - l == 2) {
        t->t_type = TREE_TYPE_BOTH_LEAF;
        t->t_left = (void *)points[l];
        t->t_right = (void *)points[r - 1];
        return 3;
    }

    if (r - l == 3) {
        t->t_type = TREE_TYPE_LEFT_LEAF;
        t->t_left = (void *)points[l];

        ssize_t right_idx = 0;
        ssize_t n_right_children =
            tree_build_aux(tree_nodes, points, &right_idx, m, r, max_idx);
        t->t_right = (void *)tree_index_to_ptr(tree_nodes, right_idx);
        return 2 + n_right_children;
    }

    t->t_type = TREE_TYPE_INNER;

    ssize_t left_idx = 0;
    ssize_t right_idx = 0;
    ssize_t n_left_children =
        tree_build_aux(tree_nodes, points, &left_idx, l, m, max_idx);
    ssize_t n_right_children =
        tree_build_aux(tree_nodes, points, &right_idx, m, r, max_idx);

    t->t_left = (void *)tree_index_to_ptr(tree_nodes, left_idx);
    t->t_right = (void *)tree_index_to_ptr(tree_nodes, right_idx);
    return 1 + n_left_children + n_right_children;
}

static inline int cmp_points(void const *a, void const *b) {
    double a_x = ((double const *)a)[0];
    double b_x = ((double const *)b)[0];
    if (a_x < b_x) {
        return -1;
    }
    if (a_x > b_x) {
        return 1;
    }
    return 0;
}

// returns the number of inner nodes (ie: tree_t structs)
//
static ssize_t tree_build(tree_t *tree_nodes, double const **points,
                          ssize_t n_points, ssize_t *max_idx) {
    *max_idx = 0;
    qsort(points, (size_t)n_points, sizeof(double *), cmp_points);
    ssize_t idx = 0;
    return tree_build_aux(tree_nodes, points, &idx, 0, n_points, max_idx);
}

static void tree_print(tree_t const *tree_nodes, ssize_t max_idx,
                       ssize_t n_tree_nodes) {
    fprintf(stdout, "%zd %zd\n", N_DIMENSIONS, n_tree_nodes);

    DBG(ssize_t actual_printed = 0;)
    for (ssize_t i = 0; i <= max_idx; ++i) {
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

        ssize_t left_idx = -1;
        ssize_t right_idx = -1;
        if (tree_has_left_leaf(t) != 0) {
            left_idx = n_tree_nodes + tree_left_node_idx(i);
            double const *left = (double const *)t->t_left;
            fprintf(stdout, "%zd -1 -1 %.6lf", left_idx, 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(stdout, " %lf", left[j]);
            }
            fputc('\n', stdout);
            DBG(actual_printed++;)
        } else {
            left_idx = tree_ptr_to_index(tree_nodes, (tree_t const *)t->t_left);
        }

        if (tree_has_right_leaf(t) != 0) {
            right_idx = n_tree_nodes + tree_right_node_idx(i);
            double const *right = (double const *)t->t_right;
            fprintf(stdout, "%zd -1 -1 %.6lf", right_idx, 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(stdout, " %lf", right[j]);
            }
            fputc('\n', stdout);
            DBG(actual_printed++;)
        } else {
            right_idx =
                tree_ptr_to_index(tree_nodes, (tree_t const *)t->t_right);
        }

        assert(t->t_left != t->t_right, "equal ptr");
        assert(left_idx != right_idx, "equal idx");
        assert(left_idx > 0, "");
        assert(right_idx > 0, "");
        assert((ssize_t)(t - tree_nodes) >= 0, "");
        fprintf(stdout, "%zd %zd %zd %.6lf", i, left_idx, right_idx,
                t->t_radius);
        for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
            fprintf(stdout, " %lf", t->t_center[j]);
        }
        fputc('\n', stdout);
        DBG(actual_printed++;)
    }
    DBG(assert(actual_printed == n_tree_nodes, "mismatch");)
}

int main(int argc, char *argv[]) {
    double const begin = omp_get_wtime();

    ssize_t n_points;
    double const **points = parse_args(argc, argv, &n_points);
    double const *point_values = points[0];
    DBG(N_POINTS = n_points;)

    // As discussed in gen_tree_points, the number of inner nodes is
    // assymptotically the number of leaves of the tree.
    //
    tree_t *tree_nodes =
        xcalloc((size_t)n_points, tree_sizeof());  // FMA initialization
    ssize_t max_idx = 0;
    ssize_t n_tree_nodes = tree_build(tree_nodes, points, n_points, &max_idx);

    fprintf(stderr, "%.1lf\n", omp_get_wtime() - begin);

    tree_print(tree_nodes, max_idx, n_tree_nodes);

    free((void *)point_values);
    free(points);
    free(tree_nodes);
}
