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

// parse the arguments
double *parse_args(int argc, char *argv[], ssize_t *n_dimensions,
                   ssize_t *n_points) {
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
    double *pt_arr =
        xmalloc(2 * sizeof(double) * (size_t)*n_dimensions * (size_t)*n_points);

    for (ssize_t i = 0; i < *n_points; i++) {
        for (ssize_t j = 0; j < *n_dimensions; j++) {
            pt_arr[i * (*n_dimensions) + j] = RANGE * ((double)random()) / RAND_MAX;
        }
    }

    return pt_arr;
}

typedef struct tree_t {
    double *t_point;  // pointer to point list
    double t_radius;
    struct tree_t *t_left;
    struct tree_t *t_right;
} tree_t;

// Safety: this assumes both pointers are to the same continuous
static inline ssize_t ptr_to_idx(tree_t const *base_ptr, tree_t const *ptr) {
    return (ptr == NULL) ? -1 : (size_t)(ptr - base_ptr) / (sizeof(tree_t));
}

static void tree_print(tree_t const *tree_nodes, ssize_t n_dimensions,
                       ssize_t n_inner_nodes) {
    for (ssize_t i = 0; i < n_inner_nodes; ++i) {
        fprintf(stdout, "%zd %zd %zd %.6lf", i,
                ptr_to_idx(tree_nodes, tree_nodes[i].t_left),
                ptr_to_idx(tree_nodes, tree_nodes[i].t_right),
                tree_nodes[i].t_radius);
        for (ssize_t j = 0; j < n_dimensions; ++j) {
            fprintf(stdout, " %lf", tree_nodes[i].t_point[j]);
        }
        fputc('\n', stdout);
    }
}

static ssize_t tree_build(tree_t *tree_nodes, double *points,
                          ssize_t n_dimensions, ssize_t n_points) {
    ssize_t const n_leaves = n_points;
    return n_points - n_leaves;
}

int main(int argc, char *argv[]) {
    double const begin = omp_get_wtime();

    ssize_t n_dimensions;
    ssize_t n_points;
    double *points = parse_args(argc, argv, &n_dimensions, &n_points);

    // as discussed in gen_tree_points, the number of inner nodes is
    // assymptotically the number of leaves of the tree
    //
    tree_t *tree_nodes = xmalloc((size_t)n_points * sizeof(tree_t));
    ssize_t n_inner_nodes =
        tree_build(tree_nodes, points, n_dimensions, n_points);

    fprintf(stderr, "%.1lf\n", omp_get_wtime() - begin);

    tree_print(tree_nodes, n_dimensions, n_inner_nodes);

    free(points);
    free(tree_nodes);
}
