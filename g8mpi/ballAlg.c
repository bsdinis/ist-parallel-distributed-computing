#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ssize_t
#define ssize_t __ssize_t
#endif

ssize_t N_DIMENSIONS = 0;

#define xmalloc(size) priv_xmalloc__(__FILE__, __LINE__, size)
#define xrealloc(ptr, size) priv_xrealloc__(__FILE__, __LINE__, ptr, size)
#define xcalloc(nmemb, size) priv_xcalloc__(__FILE__, __LINE__, nmemb, size)

#define KILL(...)                                                \
    {                                                            \
        fprintf(stderr, "[ERROR] %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
        exit(EXIT_FAILURE);                                      \
        __builtin_unreachable();                                 \
    }

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
#ifndef NDEBUG

#endif /* NDEBUG */

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

// checked malloc
//
void *priv_xmalloc__(char const *file, int lineno, size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL && size != 0) {
        fprintf(stderr, "[ERROR] %s:%d | xmalloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return ptr;
}

// checked realloc
//
void *priv_xrealloc__(char const *file, int lineno, void *ptr, size_t size) {
    void *new_ptr = realloc(ptr, size);
    if (size != 0 && new_ptr == NULL && ptr != NULL) {
        fprintf(stderr, "[ERROR] %s:%d | xrealloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return new_ptr;
}

// checked calloc
//
void *priv_xcalloc__(char const *file, int lineno, size_t nmemb, size_t size) {
    void *ptr = calloc(nmemb, size);
    if (ptr == NULL && size != 0) {
        fprintf(stderr, "[ERROR] %s:%d | xcalloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return ptr;
}

/***
 * generic geometric functions
 */

// ----------------------------------------------------------
// Distance functions
// ----------------------------------------------------------

// computes the square of the distance between to points
//
double distance_squared(double const *pt_1, double const *pt_2) {
    double d_s = 0;
    for (ssize_t i = 0; i < N_DIMENSIONS; i++) {
        double aux = (pt_1[i] - pt_2[i]);
        d_s += aux * aux;
    }
    return d_s;
}

// computes the inner product between the difference of two points and a vector
// this avoids actually constructing the difference between the two points
//
double diff_inner_product(double const *pt, double const *a,
                          double const *b_minus_a) {
    double prod = 0;
    for (ssize_t i = 0; i < N_DIMENSIONS; i++) {
        prod += b_minus_a[i] * (pt[i] - a[i]);
    }

    return prod;
}

// a . b
double inner_product(double const *a, double const *b) {
    double prod = 0;
    for (ssize_t i = 0; i < N_DIMENSIONS; i++) {
        prod += a[i] * b[i];
    }

    return prod;
}

/***
 * strategies for dividing a set of points.
 */

// a stratey receives a list of points, a range [l, r[, and out-ptrs for the
// result. it returns the distance between these two points
//
typedef double (*strategy_t)(double const **, ssize_t, ssize_t, ssize_t *,
                             ssize_t *);

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

/**
 * Functions for the algorithm
 */

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
    double *lo = &vec[l];
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

    double pivot = *mid;
    swap_double(mid, hi);  // store pivot away

    // LOG("SWAPED, l = %zd r = %zd piv = %f", l, r ,pivot);

    ssize_t i = l;
    ssize_t j = r - 2;

    while (i < j) {
        while (vec[i] < pivot && i < r - 2) {
            i += 1;
        }
        while (vec[j] > pivot && l < j) {
            j -= 1;
        }

        if (i < j) {
            swap_double(&vec[i], &vec[j]);
            i += 1;
            j -= 1;
        }
    }
    swap_double(&vec[i], &vec[r - 1]);

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

    // char line_buf[8 * 4096];

    // size_t off = 0;
    // off += sprintf(line_buf, "\nsize = %zd\nBEFORE", size);
    // for (int i = 0; i < size; ++i) {
    //     off += sprintf(line_buf + off, " %f", vec[i]);
    // }
    // off += sprintf(line_buf + off, "\n");

    double median = qselect(vec, 0, (size_t)size, k);

    // off += sprintf(line_buf + off, " AFTER");
    // for (int i = 0; i < size; ++i) {
    //     off += sprintf(line_buf + off, " %f", vec[i]);
    // }
    // off += sprintf(line_buf + off, "\n");

    // off += sprintf(line_buf + off, "median: %f\n", median);

    if (size % 2 == 0) {
        median = (median + find_max(vec, k)) / 2;
        // off += sprintf(line_buf + off, "correction: %f\n", median);
    }

    // fputs(line_buf, stderr);
    // fflush(stderr);

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
                             strategy_t find_points, double *center) {
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

    for (ssize_t i = 0; i < r - l; ++i) {
        products[i] = diff_inner_product(points[l + i], points[a], b_minus_a);
        products_aux[i] = products[i];
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

/*
 * Tree
 */

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
static inline ssize_t tree_left_node_idx(ssize_t parent) { return parent + 1; }
static inline ssize_t tree_right_node_idx(ssize_t parent, ssize_t n_points) {
    return parent + n_points / 2;
}

static inline ssize_t tree_leaf_idx(ssize_t index, ssize_t n_points) {
    return index + n_points - 1;
}

static inline ssize_t tree_leaf_idx_to_index(ssize_t idx, ssize_t n_points) {
    return idx + 1 - n_points;
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
static void tree_print(tree_t const *tree_nodes, double const **points,
                       ssize_t n_points) {
    fprintf(stdout, "%zd %zd\n", N_DIMENSIONS, 2 * n_points - 1);

    for (ssize_t i = 0; i < n_points - 1; ++i) {
        tree_t const *t = tree_index_to_ptr(tree_nodes, i);
        // LOG("printing %zd {\n"
        //     "   type:   %d,\n"
        //     "   radius: %lf,\n"
        //     "   left:   %zd,\n"
        //     "   right:  %zd,\n"
        //     "}", i, t->t_type, t->t_radius, t->t_left, t->t_right);

        if (tree_has_left_leaf(t) != 0) {
            fprintf(stdout, "%zd -1 -1 %.6lf", t->t_left, 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(stdout, " %lf",
                        points[tree_leaf_idx_to_index(t->t_left, n_points)][j]);
            }
            fputc('\n', stdout);
        }

        if (tree_has_right_leaf(t) != 0) {
            fprintf(stdout, "%zd -1 -1 %.6lf", t->t_right, 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(
                    stdout, " %lf",
                    points[tree_leaf_idx_to_index(t->t_right, n_points)][j]);
            }
            fputc('\n', stdout);
        }

        fprintf(stdout, "%zd %zd %zd %.6lf", i, t->t_left, t->t_right,
                t->t_radius);
        for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
            fprintf(stdout, " %lf", t->t_center[j]);
        }
        fputc('\n', stdout);
    }
}
#endif

static void tree_build_aux(tree_t *tree_nodes, double const **points,
                           ssize_t n_points, ssize_t idx, ssize_t l, ssize_t r,
                           strategy_t find_points) {
    assert(r - l > 1, "1-sized trees are out of scope");

    tree_t *t = tree_index_to_ptr(tree_nodes, idx);
    // LOG("building tree node %zd: %p [%zd, %zd[ -> L = %zd, R = %zd", idx,
    //(void*)t, l, r, tree_left_node_idx(idx), tree_right_node_idx(idx));

    // double const begin = omp_get_wtime();
    divide_point_set(points, l, r, find_points, t->t_center);

    t->t_radius = compute_radius(points, l, r, t->t_center);

    // fprintf(stderr, "%zd %.12lf\n", depth, omp_get_wtime() - begin);

    ssize_t m = (l + r) / 2;
    if (r - l == 2) {
        t->t_type = TREE_TYPE_BOTH_LEAF;
        t->t_left = tree_leaf_idx(l, n_points);
        t->t_right = tree_leaf_idx(r - 1, n_points);
        return;
    }

    if (r - l == 3) {
        t->t_type = TREE_TYPE_LEFT_LEAF;
        t->t_left = tree_leaf_idx(l, n_points);
        t->t_right = tree_right_node_idx(idx, 3);

        tree_build_aux(tree_nodes, points, n_points, t->t_right, m, r,
                       find_points);
        return;
    }

    t->t_type = TREE_TYPE_INNER;
    t->t_left = tree_left_node_idx(idx);
    t->t_right = tree_right_node_idx(idx, r - l);

    tree_build_aux(tree_nodes, points, n_points, t->t_left, l, m, find_points);
    tree_build_aux(tree_nodes, points, n_points, t->t_right, m, r, find_points);
}

// Compute the tree
//
static void tree_build(tree_t *tree_nodes, double const **points,
                       ssize_t n_points, strategy_t find_points) {
    omp_set_nested(1);
    tree_build_aux(tree_nodes, points, n_points, 0 /* idx */, 0 /* l */,
                   n_points /* r */, find_points /* strategy */);
}

#ifndef RANGE
#define RANGE 10
#endif  // RANGE

// parse the arguments
static double const **parse_args(int argc, char *argv[], ssize_t *n_points,
                                 tree_t **tree_nodes) {
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

    double **pt_ptr = xmalloc(sizeof(double *) * (size_t)*n_points);
    ;
    double *pt_arr =
        xmalloc(sizeof(double) * (size_t)N_DIMENSIONS * (size_t)*n_points);

    for (ssize_t i = 0; i < *n_points; i++) {
        // fprintf(stderr, "%d pt_arr fill\n", omp_get_thread_num());
        for (ssize_t j = 0; j < N_DIMENSIONS; j++) {
            pt_arr[i * (N_DIMENSIONS) + j] =
                RANGE * ((double)rand()) / RAND_MAX;
        }
    }

    for (ssize_t i = 0; i < *n_points; i++) {
        pt_ptr[i] = &pt_arr[i * (N_DIMENSIONS)];
    }
    // As discussed, the number of inner nodes is
    // at most the number of leaves of the tree.
    //
    *tree_nodes = xcalloc((size_t)(*n_points - 1),
                          tree_sizeof());  // FMA initialization

    return (double const **)pt_ptr;
}

#undef RANGE

static int strategy_main(int argc, char **argv, strategy_t strategy) {
    double const begin = omp_get_wtime();

    tree_t *tree_nodes = NULL;
    ssize_t n_points = 0;
    double const **points = parse_args(argc, argv, &n_points, &tree_nodes);
    double const *point_values = points[0];

    tree_build(tree_nodes, points, n_points, strategy);

    fprintf(stderr, "%.1lf\n", omp_get_wtime() - begin);

#ifndef PROFILE
    tree_print(tree_nodes, points, n_points);
#endif

    free((void *)point_values);
    free(points);
    free(tree_nodes);

    return 0;
}

int main(int argc, char *argv[]) {
    omp_set_num_threads(1);
    return strategy_main(argc, argv, most_distant_approx);
}
