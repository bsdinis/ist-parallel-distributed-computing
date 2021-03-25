// ballAlg
// seraial implementation
//

#include <errno.h>
#include <omp.h>
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

#define xmalloc(size) _priv_xmalloc(__FILE__, __LINE__, size)
#define xrealloc(ptr, size) _priv_xrealloc(__FILE__, __LINE__, ptr, size)

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
    TREE_TYPE_INNER      = 0,  // both left and right are inners
    TREE_TYPE_LEFT_LEAF  = 1,  // left is a leaf
    TREE_TYPE_RIGHT_LEAF = 2,  // right is a leaf // WILL NEVER OCCOUR
    TREE_TYPE_BOTH_LEAF  = 3,  // both are leaves
} tree_type_t;

typedef struct {
    tree_type_t t_type;
    double t_radius;
    void *t_left;  // tree_t or double
    void *t_right; // tree_t or double
    double t_center[];
} tree_t;

// consistent functions for assignind indices
static inline ssize_t tree_left_node_idx(ssize_t parent) {
    return 2 * parent + 1;
}
static inline ssize_t tree_right_node_idx(ssize_t parent) {
    return 2 * parent + 2;
}

static inline size_t tree_sizeof(ssize_t n_dim) {
    return sizeof(tree_t) + sizeof(double) * (size_t) n_dim;
}

static inline tree_t * tree_index_to_ptr(tree_t * tree_vec, ssize_t idx, ssize_t n_dim) {
    return (tree_t *) (((uint8_t*)tree_vec) + idx * tree_sizeof(n_dim));
}

// Safety: this assumes both pointers are to the same contiguous array
static inline ssize_t tree_ptr_to_index(tree_t const *base_ptr, tree_t const *ptr, ssize_t n_dim) {
    return (ptr == NULL) ? -1 : (ptr - base_ptr) / (tree_sizeof(n_dim));
}

// parse the arguments
static double **parse_args(int argc, char *argv[], ssize_t *n_dimensions, ssize_t *n_points) {
    if (argc != 4) {
        KILL("usage: %s <n_dimensions> <n_points> <seed>", argv[0]);
    }

    errno = 0;
    *n_dimensions = strtoll(argv[1], NULL, 0);
    if (errno != 0) {
        KILL("failed to parse %s as integer: %s", argv[1], strerror(errno));
    }
    if (*n_dimensions < 2) {
        KILL("Illegal number of dimensions (%zd), must be above 1.",
             *n_dimensions);
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
    double  *pt_arr = xmalloc(sizeof(double)   * (size_t)*n_dimensions * (size_t)*n_points);
    double **pt_ptr = xmalloc(sizeof(double *) * (size_t)*n_points);

    // double ** pt_points = xmalloc(sizeof(double*) * (size_t)*n_points );

    for (ssize_t i = 0; i < *n_points; i++) {
        for (ssize_t j = 0; j < *n_dimensions; j++) {
            pt_arr[i * (*n_dimensions) + j] = RANGE * ((double)random()) / RAND_MAX;
        }
        pt_ptr[i] = & pt_arr[i * (*n_dimensions)];
    }

    return pt_ptr;
}

// D**2 = Sum {v=1 to n_dimensions} (pt1_x - pt2_x)**2
static inline double distance_squared(ssize_t n_dimensions, double const *pt_1, double const *pt_2) {
    double d_s = 0;
    for (ssize_t i = 0; i < n_dimensions; i++) {
        double aux = (pt_1[i] - pt_2[i]);
        d_s += aux*aux;
    }
    return d_s;
}

// Returns:
//    a and b by reference (as indeces)
//    l and r define a range to search (r is exclusive)
//    distance squared
static double find_two_most_distant(
        double const ** points,
        ssize_t l,
        ssize_t r,
        ssize_t n_dimensions,
        ssize_t * a,
        ssize_t * b) {

    ssize_t max_a = 0;
    ssize_t max_b = 0;
    double long_dist = 0;

    // O(n**2)
    for (ssize_t i = l; i < r; i++) {
        for (ssize_t j = i+1; j < r; j++) {
            double aux_long_dist = distance_squared(n_dimensions, points[i], points[j]);
            if (aux_long_dist > long_dist) {
                long_dist = aux_long_dist;
                max_a = i;
                max_b = j;
            }
        }
    }

    *a = max_a;
    *b = max_b;

    return long_dist;
}

// computes (pt - a) . b_minus_a
static double inner_product(double const *a, double const *b_minus_a,
        double const *pt, ssize_t n_dimensions) {

    double projection = 0;
    for (ssize_t i = 0; i < n_dimensions ; i++) {
        projection += b_minus_a[i] * (pt[i] - a[i]);
    }

    return projection;
}

static void insertion_sort(double * vec, ssize_t l, ssize_t r) {
    for (ssize_t i = l + 1; i < r; ++i) {
        ssize_t j = i - 1;
        double  val = vec[i];
        while (j >= l && val < vec[j]) {
            vec[j + 1] = vec[j];
            --j;
        }

        vec[j + 1] = val;
    }
}

static void choose_pivots(double const *vec, ssize_t l, ssize_t r, double *pivot1, double *pivot2)  {
    // choose N values from the array [0, 1, 2, 3, 4], and consider the 33th and 66th percentile
    //
    double pivot_pool[PIVOT_POOL_N];
    for (ssize_t i = 0; i < PIVOT_POOL_N; ++i) {
        pivot_pool[i] = (double)l + random() % (r - l);
    }
    insertion_sort(pivot_pool, 0, PIVOT_POOL_N);

    *pivot1 = pivot_pool[PIVOT_POOL_N/3];
    *pivot2 = pivot_pool[PIVOT_POOL_N - PIVOT_POOL_N/3];
}

static inline void swap_double(double *a, double *b) {
    double tmp1 = *a;
    double tmp2 = *b;
    *a = tmp2;
    *b = tmp1;
}

static inline void swap_ptr(void **a, void **b) {
    void * tmp1 = *a;
    void * tmp2 = *b;
    *a = tmp2;
    *b = tmp1;
}

static void three_way_partition(double *vec, ssize_t l, ssize_t r, double pivot1, double pivot2, ssize_t *pivot1_loc, ssize_t *pivot2_loc) {
    ssize_t p1_loc = l;
    ssize_t p2_loc = r;
    for (ssize_t i = l; i < r;) {
        if (vec[i] < pivot1) {
            swap_double(&vec[i++], &vec[l++]);
        } else if (vec[i] > pivot2) {
            swap_double(&vec[i], &vec[r-- - 1]);
        } else {
            if (i > p1_loc && vec[i] == pivot1) {
                p1_loc = i;
            }
            if (i < p2_loc && vec[i] == pivot2) {
                p2_loc = i;
            }
            ++i;
        }
    }

    *pivot1_loc = p1_loc;
    *pivot2_loc = p2_loc;
}

// base 3 quicksort which computes the median in O(n)
//
// Pick 2 pivots, one in the first half and one in the second half
// Partition accordingly
//
// Repeat only in the center, until the size is small (then just sort)
//
static double find_median(double * vec, ssize_t l, ssize_t r, ssize_t orig_m1, ssize_t orig_m2) {
    if (r - l < INSERTION_SORT_CONSTANT) {
        // brute force
        insertion_sort(vec, l, r);
        return (vec[orig_m1] + vec[orig_m2])/2;
    }


    ssize_t pivot1_loc;
    ssize_t pivot2_loc;
    do {
        double pivot1;
        double pivot2;
        choose_pivots(vec, l, r, &pivot1, &pivot2);
        three_way_partition(vec, l, r, pivot1, pivot2, &pivot1_loc, &pivot2_loc);
    } while (pivot1_loc > orig_m1 - 1 || pivot2_loc < orig_m2 + 1);


    return find_median(vec, pivot1_loc, pivot2_loc, orig_m1, orig_m2);
}


// partition on median. The key is given by the product
static void partition_on_median(double const ** points, ssize_t l, ssize_t r,
        double * products, double median) {
    ssize_t i = 0;
    ssize_t j = r - l - 1;

    while (i < j) {
        while (i < j && products[i] < median) { i++; }
        while (i < j && products[j] > median) { j--; }
        if (i < j) {
            swap_ptr((void **)&points[l + i], (void **)&points[l + j]);
            i++;
            j--;
        }
    }
}


// l and r define an interval [l, r[
static void divide_point_set(double const ** points, ssize_t l, ssize_t r, ssize_t n_dimensions) {
    ssize_t a = l;
    ssize_t b = l;
    find_two_most_distant(points, l, r, n_dimensions, &a, &b);
    double * b_minus_a = xmalloc(sizeof(double) * (size_t)n_dimensions);
    for (ssize_t i = 0; i < n_dimensions; ++i) {
        b_minus_a[i] = points[b][i] - points[a][i];
    }

    double * products = xmalloc((size_t)(r - l) * sizeof(double));
    for (ssize_t i = 0; i < r - l; ++i) {
        products[i] = inner_product(points[a], b_minus_a, points[l + i], n_dimensions);
    }

    // O(n)
    double median = find_median(products, 0, r - l, (r + l - 1)/2, (r + l)/2);

    // O(n)
    partition_on_median(points, l, r, products, median);

    free(b_minus_a);
    free(products);
}

static tree_t recurssion(tree_t *tree_nodes, size_t idx, double ** points, ssize_t l, ssize_t r, ssize_t n_dimensions) {
    divide_point_set(points, l, r, n_dimensions);

    double * m_pt = (l+r)/2;

    // TODO Find center point
    // Calculate radius

    if (r - l == 2) {
        tree_nodes[idx].t_type = TREE_TYPE_BOTH_LEAF;
        tree_nodes[idx].t_left = points[l];
        tree_nodes[idx].t_right = points[r-1];
        //TODO
    }
    else if (r - l == 3) {
        tree_nodes[idx].t_type = TREE_TYPE_LEFT_LEAF;
        tree_nodes[idx].t_left = points[l];
        recurssion(tree_nodes, tree_right_node_idx(idx), points, m_pt, r,    n_dimensions);
    }
    else {
        recurssion(tree_nodes, tree_left_node_idx(idx),  points, l,    m_pt, n_dimensions);
        recurssion(tree_nodes, tree_right_node_idx(idx), points, m_pt, r,    n_dimensions);
    }

}

static ssize_t tree_build(tree_t *tree_nodes, double ** points, ssize_t n_dimensions, ssize_t n_points) {

    recurssion(tree_nodes, 0, points, 0, n_points, n_dimensions);

    ssize_t const n_leaves = n_points;
    return n_points - n_leaves;
}

static void tree_print(tree_t const *tree_nodes,
        double ** points,
        ssize_t n_dimensions,
        ssize_t n_inner_nodes) {

    for (ssize_t i = 0; i < n_inner_nodes; ++i) {
        tree_t const *t = tree_index_to_ptr(tree_nodes, i, n_dimensions);

        if ((t->t_type & TREE_TYPE_LEFT_LEAF) != 0) {
            double * left = (double *) t->t_left;
            fprintf(stdout, "%zd -1 -1 %.6lf", tree_left_node_idx(i), 0.0);
            for (size_t j = 0; j < (size_t)n_dimensions; ++j) {
                fprintf(stdout, " %lf", left[j]);
            }
            fputc('\n', stdout);
        }
        if ((t->t_type & TREE_TYPE_RIGHT_LEAF) != 0) { // WONT OCCOUR
            double * right = (double *) t->t_right;
            fprintf(stdout, "%zd -1 -1 %.6lf", tree_right_node_idx(i), 0.0);
            for (size_t j = 0; j < (size_t)n_dimensions; ++j) {
                fprintf(stdout, " %lf", right[j]);
            }
            fputc('\n', stdout);
        }

        fprintf(stdout, "%zd %zd %zd %.6lf", i, tree_left_node_idx(i), tree_right_node_idx(i), t->t_radius);
        for (size_t j = 0; j < (size_t)n_dimensions; ++j) {
            fprintf(stdout, " %lf", t->t_center[j]);
        }
        fputc('\n', stdout);
    }
}

int main(int argc, char *argv[]) {
    double const begin = omp_get_wtime();

    ssize_t n_dimensions;
    ssize_t n_points;
    double ** points = parse_args(argc, argv, &n_dimensions, &n_points);
    double * point_values = points[0];

    // as discussed in gen_tree_points, the number of inner nodes is
    // assymptotically the number of leaves of the tree
    //
    tree_t *tree_nodes = xmalloc((size_t)n_points * (sizeof(tree_t) + sizeof(double) * n_dimensions)); // FMA initialization
    ssize_t n_inner_nodes =
        tree_build(tree_nodes, points, n_dimensions, n_points);

    fprintf(stderr, "%.1lf\n", omp_get_wtime() - begin);

    tree_print(tree_nodes, points, n_dimensions, n_inner_nodes);

    free(point_values);
    free(points);
    free(tree_nodes);
}
