#include <errno.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef ssize_t
#define ssize_t __ssize_t
#endif

// TODO: delete
char line_buf[4096];

typedef enum computation_mode_t {
    // everything fits in memory, we have all nodes with the whole dataset
    CM_SINGLE_NODE,

    // a single node does not hold everything in memory
    CM_DISTRIBUTED,

    // there are other nodes that can fit everything in memory, just do nothing
    CM_PASSIVE,
} computation_mode_t;

ssize_t N_DIMENSIONS = 0;  // NOLINT

#define xmalloc(size) priv_xmalloc__(__FILE__, __LINE__, size)
#define xrealloc(ptr, size) priv_xrealloc__(__FILE__, __LINE__, ptr, size)
#define xcalloc(nmemb, size) priv_xcalloc__(__FILE__, __LINE__, nmemb, size)

#define KILL(...)                                                \
    {                                                            \
        fprintf(stderr, "[ERROR] %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
        fflush(stderr);                                          \
        MPI_Finalize();                                          \
        exit(EXIT_FAILURE);                                      \
        __builtin_unreachable();                                 \
    }

#define WARN(...)                                                \
    {                                                            \
        fprintf(stderr, "[WARN]  %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
        fflush(stderr);                                          \
    }

#define LOG(...)                                                 \
    {                                                            \
        fprintf(stderr, "[LOG]   %s:%d | ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                            \
        fputc('\n', stderr);                                     \
        fflush(stderr);                                          \
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

// Find the two most distant points approximately
// a is the furthest point from points[l]
// b is the furthes point from a
//
// time = 2*n
//
static double dist_most_distant_approx(double const **points, ssize_t l,
                                       ssize_t r, int proc_id, int n_procs,
                                       MPI_Comm comm, double *a, double *b) {
    double first_point[N_DIMENSIONS];
    size_t off = 0;
    if (proc_id == 0) {
        memcpy(first_point, points[l], sizeof(first_point));
    } else {
        memset(first_point, 0, sizeof(first_point));
    }

    MPI_Bcast(&first_point[0], N_DIMENSIONS, MPI_DOUBLE, 0, comm);

    double dist_l_a = 0;
    ssize_t a_idx = 0;
    for (ssize_t i = l; i < r; ++i) {
        double dist = distance_squared(first_point, points[i]);
        if (dist > dist_l_a) {
            dist_l_a = dist;
            a_idx = i;
        }
    }
    LOG("[%d] a0 => found %lf", proc_id, dist_l_a);

    double recv_points[n_procs][N_DIMENSIONS];
    double send_points[n_procs][N_DIMENSIONS];
    for (int idx = 0; idx < n_procs; ++idx) {
        memcpy(send_points[idx], points[a_idx], N_DIMENSIONS * sizeof(double));
    }
    MPI_Alltoall(send_points, N_DIMENSIONS, MPI_DOUBLE, recv_points,
                 N_DIMENSIONS, MPI_DOUBLE, comm);

    dist_l_a = 0;
    a_idx = 0;
    for (ssize_t i = 0; i < n_procs; ++i) {
        double dist = distance_squared(first_point, recv_points[i]);
        if (dist > dist_l_a) {
            dist_l_a = dist;
            a_idx = i;
        }
    }
    memcpy(a, recv_points[a_idx], N_DIMENSIONS * sizeof(double));
    LOG("[%d] af => found %lf", proc_id, dist_l_a);

    double dist_a_b = 0;
    ssize_t b_idx = 0;
    for (ssize_t i = l; i < r; ++i) {
        double dist = distance_squared(a, points[i]);
        if (dist > dist_a_b) {
            dist_a_b = dist;
            b_idx = i;
        }
    }
    memset(recv_points, 0, sizeof(recv_points));
    LOG("[%d] b0 => found %lf", proc_id, dist_a_b);

    for (int idx = 0; idx < n_procs; ++idx) {
        memcpy(send_points[idx], points[b_idx], N_DIMENSIONS * sizeof(double));
    }
    MPI_Alltoall(send_points, N_DIMENSIONS, MPI_DOUBLE, recv_points,
                 N_DIMENSIONS, MPI_DOUBLE, comm);

    dist_a_b = 0;
    b_idx = 0;
    for (ssize_t i = 0; i < n_procs; ++i) {
        double dist = distance_squared(a, recv_points[i]);
        if (dist > dist_a_b) {
            dist_a_b = dist;
            b_idx = i;
        }
    }
    memcpy(b, recv_points[b_idx], N_DIMENSIONS * sizeof(double));
    LOG("[%d] bf => found %lf", proc_id, dist_a_b);

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
static double find_max(double *const vec, ssize_t l, ssize_t r) {
    double max = 0.0;
    for (size_t i = l; i < r; i++) {
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
static double qselect(double *vec, size_t l, size_t r, size_t k) {  // NOLINT
    // find the partition

    size_t p = partition(vec, l, r);

    if (p == k || r - l <= 3) {
        return vec[k];
    }

    if (p > k) {
        return qselect(vec, l, p, k);
    }

    return qselect(vec, p + 1, r, k);
}

// Find the median value of a vector
//
static double find_median(double *vec, ssize_t l, ssize_t r) {
    size_t k = (size_t)(r + l) / 2;
    double median = qselect(vec, l, r, k);
    if ((r - l) % 2 == 0) {
        median = (median + find_max(vec, l, k)) / 2;
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
static void partition_on_median(double const **points, double const *products,
                                ssize_t l, ssize_t r, double median) {
    ssize_t i = l;
    ssize_t j = r - 1;
    ssize_t k = l + (r - l) / 2;
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
            swap_ptr((void **)&points[i], (void **)&points[j]);
            i++;
            j--;
        }
    }
    ssize_t m = (r + l) / 2;
    swap_ptr((void **)&points[k],
             (void **)&points[m]);  // ensure median is on the right set
}

// ----------------------------------------------------------
// Algorithm
// ----------------------------------------------------------

// Divide a point set, finding its center (for the ball algorithm)
// will reorder the points in the set.
//
static void divide_point_set(double const **points, double *inner_products,
                             double *inner_products_aux, ssize_t l, ssize_t r,
                             double *center) {
    ssize_t a = l;
    ssize_t b = l;

    // 2 * n
    // No point in parallelizing: requires too much synchronization overhead
    double dist = most_distant_approx(points, l, r, &a, &b);

    // points[a] may change after the partition
    double const *a_ptr = points[a];
    double b_minus_a[N_DIMENSIONS];
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        b_minus_a[i] = points[b][i] - points[a][i];
    }

    for (ssize_t i = l; i < r; ++i) {
        inner_products[i] = diff_inner_product(points[i], points[a], b_minus_a);
        inner_products_aux[i] = inner_products[i];
    }

    // O(n)
    // No point in parallelizing: requires too much synchronization overhead
    double median = find_median(inner_products, l, r);

    // O(n)
    // Not possible in parallelizing
    partition_on_median(points, inner_products_aux, l, r, median);

    double normalized_median = median / dist;
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        center[i] =
            a_ptr[i] + (b_minus_a[i] *       // NOLINT: this does not see that
                        normalized_median);  // we fully initialize b_minus_a
    }
}

// Divide a point set, finding its center (for the ball algorithm)
// will reorder the points in the set.
//
static void dist_divide_point_set(double const **points, double *inner_products,
                                  double *inner_products_aux, ssize_t l,
                                  ssize_t r, int proc_id, int n_procs,
                                  MPI_Comm comm, double *center) {
    // 2 * n
    double a[N_DIMENSIONS];
    double b[N_DIMENSIONS];
    double dist =
        dist_most_distant_approx(points, l, r, proc_id, n_procs, comm, a, b);

    // points[a] may change after the partition
    double b_minus_a[N_DIMENSIONS];
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        b_minus_a[i] = b[i] - a[i];
    }

    for (ssize_t i = l; i < r; ++i) {
        inner_products[i] = diff_inner_product(points[i], a, b_minus_a);
        inner_products_aux[i] = inner_products[i];
    }

    // TODO
    // O(n)
    // No point in parallelizing: requires too much synchronization overhead
    double median = find_median(inner_products, l, r);

    // O(n)
    // Not possible in parallelizing
    partition_on_median(points, inner_products_aux, l, r, median);  // TODO

    // TODO
    double normalized_median = median / dist;
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        center[i] =
            a[i] + (b_minus_a[i] *       // NOLINT: this does not see that
                    normalized_median);  // we fully initialize b_minus_a
    }
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

// Compute radius of a ball, given its center
// As this requires synchronization to compute in parallel, we have observed
// slowdowns from trying to parallelize this.
//
// Returns radius
//
static double dist_compute_radius(double const **points, ssize_t l, ssize_t r,
                                  int proc_id, int n_procs, MPI_Comm comm,
                                  double const *center) {
    double max_dist_sq = 0.0;
    for (ssize_t i = l; i < r; i++) {
        double dist = distance_squared(center, points[i]);
        if (dist > max_dist_sq) {
            max_dist_sq = dist;
        }
    }

    LOG("[%2d] r0 => %6.2lf", proc_id, max_dist_sq);

    double max_sq_mine[n_procs];
    for (ssize_t i = 0; i < n_procs; ++i) {
        max_sq_mine[i] = max_dist_sq;
    }

    double max_sq_all[n_procs];
    MPI_Gather(max_sq_mine, 1, MPI_DOUBLE, max_sq_all, 1, MPI_DOUBLE, 0, comm);

    if (proc_id == 0) {
        max_dist_sq = 0.0;
        for (ssize_t i = 0; i < n_procs; ++i) {
            if (max_dist_sq < max_sq_all[i]) {
                max_dist_sq = max_sq_all[i];
            }
        }
        LOG("[%2d] rf => %6.2lf", proc_id, max_dist_sq);

        return sqrt(max_dist_sq);
    } else {
        return 0.0;
    }
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

static void tree_build_dist_aux(
    tree_t *tree_nodes,     /* set of tree nodes */
    double const **points,  /* list of points */
    double *inner_products, /* list of inner products: preallocated to make sure
                               this function is no-alloc */
    double *inner_products_aux, /* list of auxiliar inner products: preallocated
                                   to make sure this function is no-alloc */
    ssize_t idx,                /* index of this node */
    ssize_t l,                  /* index of the left-most point for this node */
    ssize_t r,   /* index of the right-most point for this node */
    int proc_id, /* mpi proccess id in the communication group*/
    int n_procs, /* number of mpi processes active in this communication group*/
    MPI_Comm comm, /* communication group */
    ssize_t ava,   /* number of available omp threads */
    ssize_t depth  /* depth of the local computation */
) {
    assert(r - l > 1, "1-sized trees are out of scope");

    tree_t *t = tree_index_to_ptr(tree_nodes, idx);
    // LOG("building tree node %zd: %p [%zd, %zd[ -> L = %zd, R = %zd", idx,
    //(void*)t, l, r, tree_left_node_idx(idx), tree_right_node_idx(idx));

    // double const begin = omp_get_wtime();

    dist_divide_point_set(points, inner_products, inner_products_aux, l, r,
                          proc_id, n_procs, comm, t->t_center);

    double radius =
        dist_compute_radius(points, l, r, proc_id, n_procs, comm, t->t_center);
    if (proc_id == 0) {
        t->t_radius = radius;
    }

    // fprintf(stderr, "%zd %.12lf\n", depth, omp_get_wtime() - begin);
}

static void tree_build_single_aux(
    tree_t *tree_nodes,     /* set of tree nodes */
    double const **points,  /* list of points */
    double *inner_products, /* list of inner products: preallocated to make sure
                               this function is no-alloc */
    double *inner_products_aux, /* list of auxiliar inner products: preallocated
                                   to make sure this function is no-alloc */
    ssize_t idx,                /* index of this node */
    ssize_t l,                  /* index of the left-most point for this node */
    ssize_t r,    /* index of the right-most point for this node */
    int proc_id,  /* mpi proccess id */
    int n_procs,  /* number of mpi processes active in this range */
    ssize_t ava,  /* number of available omp threads */
    ssize_t depth /* depth of the local computation */
) {
    assert(r - l > 1, "1-sized trees are out of scope");

    tree_t *t = tree_index_to_ptr(tree_nodes, idx);
    // LOG("building tree node %zd: %p [%zd, %zd[ -> L = %zd, R = %zd", idx,
    //(void*)t, l, r, tree_left_node_idx(idx), tree_right_node_idx(idx));

    // double const begin = omp_get_wtime();
    divide_point_set(points, inner_products, inner_products_aux, l, r,
                     t->t_center);

    if (proc_id % n_procs == 0) {
        t->t_radius = compute_radius(points, l, r, t->t_center);
    }

    // fprintf(stderr, "%zd %.12lf\n", depth, omp_get_wtime() - begin);

    ssize_t m = (l + r) / 2;
    if (r - l == 2) {
        if (proc_id % n_procs == 0) {
            t->t_type = TREE_TYPE_BOTH_LEAF;
            t->t_left = l;
            t->t_right = r - 1;
        }
        return;
    }

    if (r - l == 3) {
        if (proc_id % n_procs == 0) {  // Just 1 of each set
            t->t_type = TREE_TYPE_LEFT_LEAF;
            t->t_left = l;
            t->t_right = tree_right_node_idx(idx);

            tree_build_single_aux(tree_nodes, points, inner_products,
                                  inner_products_aux, t->t_right, m, r, proc_id,
                                  n_procs, ava, depth + 1);
        }
        return;
    }

    t->t_type = TREE_TYPE_INNER;
    t->t_left = tree_left_node_idx(idx);
    t->t_right = tree_right_node_idx(idx);

    if (n_procs == 1) {
        if (ava > 0) {  // Parallel
#pragma omp parallel sections num_threads(2)
            {
#pragma omp section
                {
                    tree_build_single_aux(
                        tree_nodes, points, inner_products, inner_products_aux,
                        t->t_left, l, m, proc_id, n_procs,
                        ava - (1 << (size_t)depth), depth + 1);
                }

#pragma omp section
                {
                    tree_build_single_aux(tree_nodes, points, inner_products,
                                          inner_products_aux, t->t_right, m, r,
                                          proc_id, n_procs, ava - (1 << depth),
                                          depth + 1);
                }
            }
        } else {  // Serial
            tree_build_single_aux(tree_nodes, points, inner_products,
                                  inner_products_aux, t->t_left, l, m, proc_id,
                                  n_procs, 0, depth + 1);
            tree_build_single_aux(tree_nodes, points, inner_products,
                                  inner_products_aux, t->t_right, m, r, proc_id,
                                  n_procs, 0, depth + 1);
        }

    } else if (proc_id % n_procs < n_procs / 2) {
        tree_build_single_aux(tree_nodes, points, inner_products,
                              inner_products_aux, t->t_left, l, m, proc_id,
                              n_procs / 2, ava, 0);
    } else if (proc_id % n_procs >= n_procs / 2) {
        tree_build_single_aux(tree_nodes, points, inner_products,
                              inner_products_aux, t->t_right, m, r, proc_id,
                              (n_procs + 1) / 2, ava, 0);
    }
}

// Compute the tree
//
static void tree_build_single(tree_t *tree_nodes, double const **points,
                              double *inner_products, ssize_t n_points,
                              int proc_id, int n_procs) {
    omp_set_nested(1);
    tree_build_single_aux(
        tree_nodes, points, inner_products, inner_products + n_points,
        0 /* idx */, 0 /* l */, n_points /* r */, proc_id, n_procs,
        omp_get_max_threads() - 1 /* available threads */, 0 /* depth */);
}

// Compute the tree
//
static void tree_build_dist(tree_t *tree_nodes, double const **points,
                            double *inner_products, ssize_t n_points,
                            ssize_t n_local_points, int proc_id, int n_procs) {
    omp_set_nested(1);
    tree_build_dist_aux(
        tree_nodes, points, inner_products, inner_products + n_local_points,
        0 /* idx */, 0 /* l */, n_local_points /* r */, proc_id, n_procs,
        MPI_COMM_WORLD, omp_get_max_threads() - 1 /* available threads */,
        0 /* depth */);
}

#ifndef PROFILE
static void tree_print(tree_t const *tree_nodes, ssize_t tree_size,
                       double const **points, int proc_id) {
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
            fflush(stdout);
        }

        if (tree_has_right_leaf(t) != 0) {
            fprintf(stdout, "%zd -1 -1 %.6lf", tree_right_node_idx(i), 0.0);
            for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
                fprintf(stdout, " %lf", points[t->t_right][j]);
            }
            fputc('\n', stdout);
            fflush(stdout);
        }

        fprintf(stdout, "%zd %zd %zd %.6lf", i, tree_left_node_idx(i),
                tree_right_node_idx(i), t->t_radius);
        for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
            fprintf(stdout, " %lf", t->t_center[j]);
        }
        fputc('\n', stdout);
        fflush(stdout);
    }
}
#endif

#ifndef RANGE
#define RANGE 10
#endif  // RANGE

static double const **allocate(int *proc_id, int *n_procs, ssize_t n_points,
                               ssize_t *n_local_points, uint32_t seed,
                               tree_t **tree_nodes, double **inner_products,
                               computation_mode_t *c_mode) {
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

    // We'll try to fit everything on the same machine.
    // If it fails, we only need n_points / n_procs
    *c_mode = CM_SINGLE_NODE;
    size_t sz_to_alloc = (size_t)n_points;

    if (true) {
        *c_mode = CM_DISTRIBUTED;
        sz_to_alloc = (size_t)n_points / *n_procs;
    }

    srand(seed);

    double **pt_ptr = malloc(sizeof(double *) * sz_to_alloc);
    if (pt_ptr == NULL) {
        if (errno != ENOMEM) {
            KILL("Failed to allocate (and not due to lack of memory");
        }

        *c_mode = CM_DISTRIBUTED;
        sz_to_alloc = ((size_t)n_points) / *n_procs;
        pt_ptr = xmalloc(sizeof(double *) * sz_to_alloc);
    }

    double *pt_arr =
        malloc(sizeof(double) * (size_t)N_DIMENSIONS * sz_to_alloc);
    if (pt_arr == NULL) {
        if (errno != ENOMEM) {
            KILL("Failed to allocate (and not due to lack of memory");
        } else if (*c_mode == CM_DISTRIBUTED) {
            KILL(
                "Even if we only allocate divided by the number of machines, "
                "we still OOM-out");
        }

        *c_mode = CM_DISTRIBUTED;
        sz_to_alloc = ((size_t)n_points) / *n_procs;
        pt_ptr = xrealloc(pt_ptr, sizeof(double *) * sz_to_alloc);
        pt_arr = xmalloc(sizeof(double) * (size_t)N_DIMENSIONS * sz_to_alloc);
    }

    *tree_nodes = calloc((size_t)(2 * sz_to_alloc), tree_sizeof());
    if (*tree_nodes == NULL) {
        if (errno != ENOMEM) {
            KILL("Failed to allocate (and not due to lack of memory");
        } else if (*c_mode == CM_DISTRIBUTED) {
            KILL(
                "Even if we only allocate divided by the number of machines, "
                "we still OOM-out");
        }

        *c_mode = CM_DISTRIBUTED;
        sz_to_alloc = ((size_t)n_points) / *n_procs;
        pt_ptr = xrealloc(pt_arr, sizeof(double *) * sz_to_alloc);
        pt_arr = xrealloc(pt_arr,
                          sizeof(double) * (size_t)N_DIMENSIONS * sz_to_alloc);
        *tree_nodes = xcalloc((size_t)(2 * sz_to_alloc), tree_sizeof());
    }

    // We need 2x the number of nodes for inner products
    // This is because we compute the median in the first half,
    // and we need the second half to remain still so that we can
    // then re-order the nodes
    *inner_products = calloc((size_t)(2 * sz_to_alloc), sizeof(double));
    if (*inner_products == NULL) {
        if (errno != ENOMEM) {
            KILL("Failed to allocate (and not due to lack of memory");
        } else if (*c_mode == CM_DISTRIBUTED) {
            KILL(
                "Even if we only allocate divided by the number of machines, "
                "we still OOM-out");
        }

        *c_mode = CM_DISTRIBUTED;
        sz_to_alloc = ((size_t)n_points) / *n_procs;
        pt_ptr = xrealloc(pt_arr, sizeof(double *) * sz_to_alloc);
        pt_arr = xrealloc(pt_arr,
                          sizeof(double) * (size_t)N_DIMENSIONS * sz_to_alloc);
        *tree_nodes =
            xrealloc(*tree_nodes, (size_t)(2 * sz_to_alloc) * tree_sizeof());
        *inner_products = xcalloc((size_t)(2 * sz_to_alloc), sizeof(double));
        memset(tree_nodes, 0, (size_t)(2 * sz_to_alloc) * tree_sizeof());
    }

    /***
     * - Some distributed, others not: abort (we could cycle until we stabilized
     *on a set of distributed nodes, which may not exist)
     * - All distributed: do all distributedly
     * - 1+ single: drop all non-singles, compute as single
     ***/

    bool mine[*n_procs];
    memset(mine, *c_mode == CM_SINGLE_NODE, sizeof(mine));
    bool single[*n_procs];
    memset(single, false, sizeof(single));
    MPI_Alltoall(mine, 1, MPI_C_BOOL, single, 1, MPI_C_BOOL, MPI_COMM_WORLD);

    int id_off = 0;
    int n_off = 0;
    for (int i = 0; i < *n_procs; i++) {
        if (!single[i]) {
            if (i < *proc_id) {
                id_off += 1;
            }
            n_off += 1;
        }
    }

    if (n_off < *n_procs) {
        if (*c_mode == CM_DISTRIBUTED) {
            *c_mode = CM_PASSIVE;
            free(pt_arr);
            free(pt_ptr);
            free(*tree_nodes);
            free(*inner_products);
            *proc_id = -1;
        } else {
            *proc_id -= id_off;
        }
        *n_procs -= n_off;
    }

    *n_local_points = (size_t)sz_to_alloc;
    if (*c_mode == CM_SINGLE_NODE) {
        // As discussed, the number of inner nodes is
        // at most the number of leaves of the tree.
        //
        for (ssize_t i = 0; i < n_points; i++) {
            // fprintf(stderr, "%d pt_arr fill\n", omp_get_thread_num());
            for (ssize_t j = 0; j < N_DIMENSIONS; j++) {
                pt_arr[i * (N_DIMENSIONS) + j] =
                    RANGE * ((double)rand()) / RAND_MAX;
            }
            pt_ptr[i] = &pt_arr[i * (N_DIMENSIONS)];
        }
    } else if (*c_mode == CM_DISTRIBUTED) {
        for (ssize_t i = 0, idx = 0; i < n_points; ++i) {
            if (i / sz_to_alloc == *proc_id) {
                for (ssize_t j = 0; j < N_DIMENSIONS; j++) {
                    double r = rand();
                    pt_arr[idx * (N_DIMENSIONS) + j] = (RANGE * r) / RAND_MAX;
                }
                pt_ptr[idx] = &pt_arr[idx * (N_DIMENSIONS)];
                idx += 1;
            } else {
                for (ssize_t j = 0; j < N_DIMENSIONS; j++) {
                    double r = rand();
                }
            }
        }
    }

    return (double const **)pt_ptr;
}

#undef RANGE

// parse the arguments
static void parse_args(int argc, char *argv[], ssize_t *n_points,
                       uint32_t *seed) {
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
    *seed = (uint32_t)strtoul(argv[3], NULL, 0);
    if (errno != 0) {
        KILL("failed to parse %s as integer: %s", argv[3], strerror(errno));
    }
}

int main(int argc, char **argv) {
    double const begin = MPI_Wtime();
    MPI_Init(&argc, &argv);

    int proc_id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    int n_procs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    uint32_t seed = 0;
    ssize_t n_points = 0;
    ssize_t n_local_points = 0;
    parse_args(argc, argv, &n_points, &seed);

    tree_t *tree_nodes = NULL;
    double *inner_products = NULL;
    computation_mode_t c_mode = CM_SINGLE_NODE;
    int old_id = proc_id;
    double const **points =
        allocate(&proc_id, &n_procs, n_points, &n_local_points, seed,
                 &tree_nodes, &inner_products, &c_mode);
    double const *point_values = points[0];

    MPI_Barrier(MPI_COMM_WORLD);
    switch (c_mode) {
        case CM_SINGLE_NODE:
            tree_build_single(tree_nodes, points, inner_products, n_points,
                              proc_id, n_procs);

            break;
        case CM_DISTRIBUTED:
            tree_build_dist(tree_nodes, points, inner_products, n_points,
                            n_local_points, proc_id, n_procs);
            break;
        case CM_PASSIVE:
            LOG("passive mode %d->%d", old_id, proc_id);
            break;
        default:
            __builtin_unreachable();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_id == 0) {
        fprintf(stderr, "%.1lf\n", MPI_Wtime() - begin);
        fflush(stderr);
#ifndef PROFILE
        fprintf(stdout, "%zd %zd\n", N_DIMENSIONS, 2 * n_points - 1);
        fflush(stdout);
        tree_print(tree_nodes, 2 * n_points, points, proc_id);
        fflush(stdout);
#endif
    }

#ifndef PROFILE
    for (int pid = 1; pid < n_procs; ++pid) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (proc_id == pid) {
            tree_print(tree_nodes, 2 * n_points, points, proc_id);
        }
    }
#endif
    MPI_Finalize();

    if (proc_id >= 0) {
        free((void *)point_values);
        free(points);
        free(inner_products);
        free(tree_nodes);
    }

    return 0;
}
