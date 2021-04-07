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

static ssize_t N_DIMENSIONS = 0;
static ssize_t N_POINTS = 0;

#define KILL(...)                                                \
    {                                                            \
        fprintf(stderr, "[ERROR] %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
        exit(EXIT_FAILURE);                                      \
        __builtin_unreachable();                                 \
    }

#ifndef NDEBUG

#define WARN(...)                                                \
    {                                                            \
        fprintf(stderr, "[WARN]  %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
    }

#define LOG(...)                                                 \
    {                                                            \
        fprintf(stderr, "[LOG]   %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
    }

#endif /* NDEBUG */

/* util functions for memory management
 */

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

static inline bool tree_is_inner(tree_t const *t) { return t->t_type == 0; }
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

// Safety: this assumes both pointers are to the same contiguous array
//         and that base_ptr < ptr
static inline ssize_t tree_ptr_to_index(tree_t const *base_ptr,
                                        tree_t const *ptr) {
    return (ptr == NULL)
               ? -1
               : (ssize_t)((size_t)(ptr - base_ptr) / (tree_sizeof()));
}

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

// Returns:
//    a and b by reference (as indeces)
//    l and r define a range to search (r is exclusive)
//    distance squared
static double find_two_most_distant(double const **points, ssize_t l, ssize_t r,
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

// computes (pt - a) . b_minus_a
static double diff_inner_product(double const *pt, double const *a,
                                 double const *b_minus_a) {
    double prod = 0;
    for (ssize_t i = 0; i < N_DIMENSIONS; i++) {
        prod += b_minus_a[i] * (pt[i] - a[i]);
    }

    return prod;
}

// a . b
static double inner_product(double const *a, double const *b) {
    double prod = 0;
    for (ssize_t i = 0; i < N_DIMENSIONS; i++) {
        prod += a[i] * b[i];
    }

    return prod;
}

static int cmp_double(void const *a, void const *b) {
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

// Find the median value of a vector
static double find_median(double *vec, ssize_t size) {
    qsort(vec, (size_t)size, sizeof(double), cmp_double);
    return (size % 2 == 0) ? (vec[(size - 2) / 2] + vec[size / 2]) / 2
                           : vec[(size - 1) / 2];
}

static inline void swap_ptr(void **a, void **b) {
    void *tmp1 = *a;
    void *tmp2 = *b;
    *a = tmp2;
    *b = tmp1;
}

// partition on median. The key is given by the product
static void partition_on_median(double const **points, ssize_t l, ssize_t r,
                                double const *products, double median) {
    ssize_t i = 0;
    ssize_t j = r - l - 1;
    LOG("left %ld, right %ld", l, r);
    while (i < j) {
        for(ssize_t k = 0; k < r - l; k++)
            fprintf(stderr, "%f ", products[k]);
        fprintf(stderr, "\n");
        while (i < j && products[i] < median) {
            i++;
        }
        while (i < j && products[j] > median) {
            j--;
        }
        if (i < j) {
            swap_ptr((void **)&points[l + i], (void **)&points[l + j]);
            i++;
            j--;
        }
    }
    LOG("After while");
    for(ssize_t k = 0; k < r - l; k++)
            fprintf(stderr, "%f ", products[k]);
        fprintf(stderr, "\n");
}

// Partition a set of points, finding its center
// l and r define an interval [l, r[
//
static void divide_point_set(double const **points, ssize_t l, ssize_t r,
                             double *center) {
    ssize_t a = l;
    ssize_t b = l;
    double dist = find_two_most_distant(points, l, r, &a, &b);

    double const *a_ptr =
        points[a];  // points[a] may change after the partition
    double *b_minus_a = xmalloc((size_t)N_DIMENSIONS * sizeof(double));
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        b_minus_a[i] = points[b][i] - points[a][i];
    }

    double *products = xmalloc((size_t)(r - l) * sizeof(double));
    double *products_aux = xmalloc((size_t)(r - l) * sizeof(double)); // OPTIMIZE only one malloc
    for (ssize_t i = 0; i < r - l; ++i) {
        products[i] = diff_inner_product(points[l + i], points[a], b_minus_a); // Optimize
        products_aux[i] = diff_inner_product(points[l + i], points[a], b_minus_a);
    }

    // O(n)
    double median = find_median(products, (r - l));

    // O(n)
    if (r - l != 3) {
        partition_on_median(points, l, r, products_aux, median);
    }
    else{
        if(products_aux[0] == median){
            double tmp1 = products_aux[0];
            double tmp2 = products_aux[1];
            products_aux[0] = tmp2;
            products_aux[1] = tmp1;
        }
        else if(products_aux[2] == median){
            double tmp1 = products_aux[2];
            double tmp2 = products_aux[1];
            products_aux[2] = tmp2;
            products_aux[1] = tmp1;
        }
    }

    double normalized_median = median / dist;
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        center[i] =
            a_ptr[i] + (b_minus_a[i] *       // NOLINT: this does not see that
                        normalized_median);  // we fully initialize b_minus_a
    }

    free(b_minus_a);
    free(products);
    free(products_aux);
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
                              ssize_t idx, ssize_t l, ssize_t r,
                              ssize_t *max_idx) {
    assert(r - l > 1, "1-sized trees are out of scope");
    assert(idx < 2 * N_POINTS,
           "there can never be more inner nodes than leaves");
    if (idx > *max_idx) {
        *max_idx = idx;
    }

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
        return 1;
    }

    if (r - l == 3) {
        t->t_type = TREE_TYPE_LEFT_LEAF;
        t->t_left = (void *)points[l];
        t->t_right =
            (void *)tree_index_to_ptr(tree_nodes, tree_right_node_idx(idx));
        return 1 + tree_build_aux(tree_nodes, points, tree_right_node_idx(idx),
                                  m, r, max_idx);
    }

    t->t_type = TREE_TYPE_INNER;
    t->t_left = (void *)tree_index_to_ptr(tree_nodes, tree_left_node_idx(idx));
    t->t_right =
        (void *)tree_index_to_ptr(tree_nodes, tree_right_node_idx(idx));
    return 1 +
           tree_build_aux(tree_nodes, points, tree_left_node_idx(idx), l, m,
                          max_idx) +
           tree_build_aux(tree_nodes, points, tree_right_node_idx(idx), m, r,
                          max_idx);
}

// returns the number of inner nodes (ie: tree_t structs)
//
static ssize_t tree_build(tree_t *tree_nodes, double const **points,
                          ssize_t n_points, ssize_t *max_idx) {
    return tree_build_aux(tree_nodes, points, 0, 0, n_points, max_idx);
}

#ifndef PROFILE
static void tree_print(tree_t const *tree_nodes, ssize_t max_idx,
                       ssize_t n_tree_nodes, ssize_t n_points) {
    fprintf(stdout, "%zd %zd\n", N_DIMENSIONS, n_tree_nodes + n_points);

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

int main(int argc, char *argv[]) {
    double const begin = omp_get_wtime();

    ssize_t n_points;
    double const **points = parse_args(argc, argv, &n_points);
    double const *point_values = points[0];
    N_POINTS = n_points;

    // As discussed in gen_tree_points, the number of inner nodes is
    // assymptotically the number of leaves of the tree
    // However, since our id assignment is sparse, we double that size, because
    // even the sparse assignment is bounded by 2*n_points
    //
    tree_t *tree_nodes =
        xcalloc(2 * (size_t)n_points, tree_sizeof());  // FMA initialization
    ssize_t max_idx = 0;

#ifndef PROFILE
    ssize_t n_tree_nodes = tree_build(tree_nodes, points, n_points, &max_idx);
#else
    tree_build(tree_nodes, points, n_points, &max_idx);
#endif

    fprintf(stderr, "%.1lf\n", omp_get_wtime() - begin);

#ifndef PROFILE
    tree_print(tree_nodes, max_idx, n_tree_nodes, n_points);
#endif

    free((void *)point_values);
    free(points);
    free(tree_nodes);
}