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

// Variable used to force distributed mode
bool DISTRIBUTED = false;

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
static double most_distant_approx(double **points, ssize_t l, ssize_t r,
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

// Find the two most distant points approximately in a distributed vector
// first_point is the first point in the "first" proccess of the set (proc_id ==
// 0) a is the furthest point from first_point b is the furthes point from a
//
static double dist_most_distant_approx(double **points, ssize_t n_local_points,
                                       int proc_id, int n_procs, MPI_Comm comm,
                                       double *a, double *b) {
    double first_point[N_DIMENSIONS];
    if (proc_id == 0) {
        memcpy(first_point, points[0], sizeof(first_point));
    } else {
        memset(first_point, 0, sizeof(first_point));
    }

    // Broadcast the first_point of the set of proccess
    MPI_Bcast(first_point, N_DIMENSIONS, MPI_DOUBLE, 0, comm);

    // Calculate local 'a'
    double dist_l_a = 0;
    ssize_t a_idx = 0;
    for (ssize_t i = 0; i < n_local_points; ++i) {
        double dist = distance_squared(first_point, points[i]);
        if (dist > dist_l_a) {
            dist_l_a = dist;
            a_idx = i;
        }
    }

    double recv_points[n_procs][N_DIMENSIONS];
    double send_points[n_procs][N_DIMENSIONS];
    for (int idx = 0; idx < n_procs; ++idx) {
        memcpy(send_points[idx], points[a_idx], N_DIMENSIONS * sizeof(double));
    }

    // Send & Recv the local 'a' of each proccess
    MPI_Alltoall(send_points, N_DIMENSIONS, MPI_DOUBLE, recv_points,
                 N_DIMENSIONS, MPI_DOUBLE, comm);

    // Calcuta the 'a' point, which is the max of locals 'a'
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

    // Calculate local 'b'
    double dist_a_b = 0;
    ssize_t b_idx = 0;
    for (ssize_t i = 0; i < n_local_points; ++i) {
        double dist = distance_squared(a, points[i]);
        if (dist > dist_a_b) {
            dist_a_b = dist;
            b_idx = i;
        }
    }
    memset(recv_points, 0, sizeof(recv_points));

    for (int idx = 0; idx < n_procs; ++idx) {
        memcpy(send_points[idx], points[b_idx], N_DIMENSIONS * sizeof(double));
    }

    // Send & Recv the local 'b' of each proccess
    MPI_Alltoall(send_points, N_DIMENSIONS, MPI_DOUBLE, recv_points,
                 N_DIMENSIONS, MPI_DOUBLE, comm);

    // Calcuta the 'b' point, which is the max of locals 'b'
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

// Chooses a pivot
// which is the median of 3 double (first, middle, last of the vector)
// The double are left in place but ordered between themselves
//
static double choose_pivot(double *vec, size_t l, size_t r) {
    double *lo = &vec[l];
    double *hi = &vec[r - 1];
    double *mid = &vec[(l + r) / 2];

    if (r - l <= 1) {
        return *lo;
    }

    if (r - l == 2) {
        if (*hi < *lo) {
            swap_double(hi, lo);
        }
        return *hi;
    }

    // Picks pivot from 3 doubles
    // leaves the 3 doubles ordered
    if (*mid < *lo) {
        swap_double(mid, lo);
    }
    if (*hi < *mid) {
        swap_double(mid, hi);
        if (*mid < *lo) {
            swap_double(mid, lo);
        }
    }

    return *mid;
}

// Partitions the vector based on the pivot
//
static size_t partition(double *vec, size_t l, size_t r, double pivot) {
    if (r - l <= 3) {
        choose_pivot(vec, l, r);  // Just order the 3 elements
        size_t i = l;
        for (; i < r - 1 && vec[i] < pivot; ++i) {   // find where pivot is
        }
        return i;
    }

    bool swapped = false;
    if (vec[(r + l) / 2] == pivot) {  // stores pivot away if in vector
        swap_double(&vec[(r + l) / 2], &vec[r - 1]);
        swapped = true;
    }

    ssize_t i = l;
    ssize_t j = (swapped) ? r - 2 : r - 1;
    ssize_t pivot_index = j;

    while (i < j) {
        while (vec[i] < pivot && i < pivot_index) {
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

    if (swapped) {
        // places pivot in position i
        // such the forall ii where (i < ii < r) vec[ii] > vec[i]
        swap_double(&vec[i], &vec[r - 1]);
    }

    return i;
}

// QuickSelect algorithm
// Finds the kth_smallest index in array
//
static double qselect(double *vec, size_t l, size_t r, size_t k) {  // NOLINT
    // find the partition

    size_t p = partition(vec, l, r, choose_pivot(vec, l, r));

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

    if ((r - l) % 2 == 0) {  // if r - l is even, median is the average
        median = (median + find_max(vec, l, k)) / 2;
    }

    return median;
}

// Distributed variant QuickSelect algorithm
// Finds the kth_smallest index in a distributed vector
//
static double dist_qselect(double *vec, ssize_t l, ssize_t r,
                           ssize_t *median_idx, ssize_t k, int proc_id,
                           int n_procs, int round, MPI_Comm comm) {
    double pivot[2];
    int leader_id = round % n_procs;  // we do not know where the median is, we
                                      // need to circulate to find it

    if (proc_id == leader_id) {
        if (r == l) {  // active vector is empty
            pivot[0] = 0.0;
            pivot[1] = 1.0;
        } else {
            pivot[0] = vec[l + (round % (r - l))];
            pivot[1] = 0.0;
        }
    }

    // Broadcast the pivot to use
    MPI_Bcast(&pivot, 2, MPI_DOUBLE, leader_id, comm);

    if (pivot[1] != 0.0) {  // No pivot // edge case
        return dist_qselect(vec, l, r, median_idx, k, proc_id, n_procs,
                            round + 1, comm);
    }

    unsigned long p = (unsigned long)partition(vec, l, r, pivot[0]);

    unsigned long sum_p = 0;
    MPI_Allreduce(&p, &sum_p, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    assert(p <= sum_p, "the sum was incorrectly computed");

    *median_idx = p;
    if (sum_p == k) {
        return pivot[0];
    }

    if (proc_id == leader_id) {  // vec[p] == pivot -> can be exclusive because
                                 // it isn't median
        if (sum_p > k) {
            return dist_qselect(vec, l, p, median_idx, k, proc_id, n_procs,
                                round + 1, comm);
        }
        return dist_qselect(vec, p + 1, r, median_idx, k, proc_id, n_procs,
                            round + 1, comm);
    }

    if (sum_p > k) {  // vec[p] != pivot -> still needs to
                      // be considerad for median
        p = (l == r) ? p
                     : p + 1;  // edge case // if (l == r) then p == r so
                               // (p + 1 > r) which would increase the vector size
        return dist_qselect(vec, l, p, median_idx, k, proc_id, n_procs,
                            round + 1, comm);
    }
    return dist_qselect(vec, p, r, median_idx, k, proc_id, n_procs, round + 1,
                        comm);
}

// Finds the max double in a distributed vector
//
static double dist_find_max(double *const vec, ssize_t size, double median,
                            int proc_id, int n_procs, MPI_Comm comm) {
    double max = 0.0;
    for (size_t i = 0; i < size && vec[i] < median; i++) {
        if (vec[i] > max) {
            max = vec[i];
        }
    }
    double g_max = 0;
    MPI_Allreduce(&max, &g_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    return g_max;
}

// Find the median value of a distributed vector
//
static double dist_find_median(double *vec, ssize_t n_local_points,
                               ssize_t *median_idx, ssize_t n_active_points,
                               int proc_id, int n_procs, MPI_Comm comm) {
    size_t k = n_active_points / 2;

    double median = dist_qselect(vec, 0, n_local_points, median_idx, k, proc_id,
                                 n_procs, 0, comm);

    if (n_active_points % 2 == 0) {
        median = (median + dist_find_max(vec, n_local_points, median, proc_id,
                                         n_procs, comm)) /
                 2;
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
static void partition_on_median(double **points, double const *products,
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

// Makes sure id the point pointer is to the left of the index then so is the
// point Forall 0 <= i < index, then (points[i] - points_values) < index *
// N_DIMENSIONS
//
static void untangle_at(double **points, double *points_values, ssize_t size,
                        ssize_t index) {
    ssize_t i = 0;
    ssize_t j = size - 1;
    double const *index_ptr = points_values + (index * N_DIMENSIONS);
    while (i < index && j >= index) {
        while (points[i] < index_ptr && i < index) {
            i += 1;
        }
        while (points[j] >= index_ptr && j >= index) {
            j -= 1;
        }
        if (i < j) {
            for (size_t dim = 0; dim < N_DIMENSIONS; ++dim) {
                swap_double(&points[i][dim], &points[j][dim]);  // NOLINT
            }
            i += 1;
            j -= 1;
        }
    }
}

// Partition the distributed vector
// Each node receives the information pertaining to how much
// each node receives.
// Each node then can autonomously compute which ranges it has
// to swap with which processes
//
static void dist_partition_on_index(double *points_values, ssize_t size,
                                    ssize_t index, int proc_id, int n_procs,
                                    MPI_Comm comm) {
    int group = (proc_id < n_procs / 2) ? 0 : 1;

    unsigned long my_index[n_procs][2];
    for (int i = 0; i < n_procs; i++) {
        if (group == 0) {
            my_index[i][0] = index;
            my_index[i][1] = size;
        } else {
            my_index[i][0] = 0;
            my_index[i][1] = index;
        }
    }
    unsigned long indeces[n_procs][2];
    memset(indeces, 0, sizeof(indeces));
    // Send and reciev how much each nodes needs to recieve
    MPI_Alltoall(my_index, 2, MPI_UNSIGNED_LONG, indeces, 2, MPI_UNSIGNED_LONG,
                 comm);

    // send_index[i][j]: how much i will send to j
    size_t send_table[n_procs][n_procs];
    memset(send_table, 0, sizeof(send_table));
    for (int i = 0; i < n_procs; ++i) {
        int gi = (i < n_procs / 2) ? 0 : 1;
        for (int j = i + 1; j < n_procs; ++j) {
            int gj = (j < n_procs / 2) ? 0 : 1;
            if (gi == gj) {
                continue;
            }
            if (indeces[i][0] == indeces[i][1]) {
                continue;
            }
            if (indeces[j][0] == indeces[j][1]) {
                continue;
            }

            size_t i_size = indeces[i][1] - indeces[i][0];
            size_t j_size = indeces[j][1] - indeces[j][0];

            send_table[i][j] = (i_size < j_size) ? i_size : j_size;
            send_table[j][i] = send_table[i][j];
            indeces[i][0] += send_table[i][j];
            indeces[j][0] += send_table[i][j];
        }
    }

    // Swaping fase (in place)
    size_t offset = (group == 0) ? index : 0;
    for (int i = 0; i < n_procs; ++i) {
        size_t size = send_table[proc_id][i];
        if (size == 0) {
            continue;
        }
        MPI_Status status;
        MPI_Sendrecv_replace(points_values + offset * N_DIMENSIONS,
                             N_DIMENSIONS * size, MPI_DOUBLE, i,
                             proc_id * n_procs + i, i, i * n_procs + proc_id,
                             comm, &status);
        offset += size;
    }
}

// ----------------------------------------------------------
// Algorithm
// ----------------------------------------------------------

// Divide a point set, finding its center (for the ball algorithm)
// will reorder the points in the set.
//
static void divide_point_set(double **points, double *inner_products,
                             double *inner_products_aux, ssize_t l, ssize_t r,
                             double *center, ssize_t omp_available) {
    ssize_t a = l;
    ssize_t b = l;

    // 2 * n
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

    // n
    if (omp_available > 1) {
#pragma omp parallel for num_threads(omp_available)
        for (ssize_t i = l; i < r; ++i) {
            inner_products[i] =
                diff_inner_product(points[i], points[a], b_minus_a);
            inner_products_aux[i] = inner_products[i];
        }
    } else {
        for (ssize_t i = l; i < r; ++i) {
            inner_products[i] =
                diff_inner_product(points[i], points[a], b_minus_a);
            inner_products_aux[i] = inner_products[i];
        }
    }

    // O(n)
    double median = find_median(inner_products, l, r);

    // O(n)
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
static void dist_divide_point_set(double **points, double *points_values,
                                  double *inner_products,
                                  double *inner_products_aux,
                                  ssize_t n_local_points,
                                  ssize_t n_active_points, int proc_id,
                                  int n_procs, MPI_Comm comm, double *center) {
    // 2 * n
    double a[N_DIMENSIONS];
    double b[N_DIMENSIONS];
    double dist = dist_most_distant_approx(points, n_local_points, proc_id,
                                           n_procs, comm, a, b);

    // points[a] may change after the partition
    double b_minus_a[N_DIMENSIONS];
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        b_minus_a[i] = b[i] - a[i];
    }

#pragma omp parallel for
    for (ssize_t i = 0; i < n_local_points; ++i) {
        inner_products[i] = diff_inner_product(points[i], a, b_minus_a);
        inner_products_aux[i] = inner_products[i];
    }

    // O(n)
    ssize_t median_idx = 0;
    double median =
        dist_find_median(inner_products, n_local_points, &median_idx,
                         n_active_points, proc_id, n_procs, comm);

    // O(n)
    partition_on_median(points, inner_products_aux, 0, n_local_points, median);

    // O(n)
    untangle_at(points, points_values, n_local_points, median_idx);

    dist_partition_on_index(points_values, n_local_points, median_idx, proc_id,
                            n_procs, comm);

    double normalized_median = median / dist;
    for (ssize_t i = 0; i < N_DIMENSIONS; ++i) {
        center[i] =
            a[i] + (b_minus_a[i] *       // NOLINT: this does not see that
                    normalized_median);  // we fully initialize b_minus_a
    }
}

// Compute radius of a ball, given its center
//
// Returns radius
//
static double compute_radius(double **points, ssize_t l, ssize_t r,
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

// Compute radius of a ball, given its center in the distributed set of points
//
// Returns radius
//
static double dist_compute_radius(double **points, ssize_t n_local_points,
                                  int proc_id, int n_procs, MPI_Comm comm,
                                  double const *center) {
    // Calulate the local radius
    double max_dist_sq = 0.0;
    for (ssize_t i = 0; i < n_local_points; i++) {
        double dist = distance_squared(center, points[i]);
        if (dist > max_dist_sq) {
            max_dist_sq = dist;
        }
    }

    // Master proccess of set calculates global radius
    double global_max = 0;
    MPI_Reduce(&max_dist_sq, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (proc_id == 0) {
        return sqrt(global_max);
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
static inline ssize_t tree_left_node_idx(ssize_t parent) { return parent + 1; }
static inline ssize_t tree_right_node_idx(ssize_t parent, ssize_t n_points) {
    return parent + n_points / 2;
}

// given an index , get the id
static inline ssize_t tree_leaf_idx_to_id(ssize_t index, ssize_t n_points,
                                          ssize_t off) {
    return index + off + n_points;
}

// given an id , get the index
static inline ssize_t tree_leaf_id_to_idx(ssize_t id, ssize_t n_points,
                                          ssize_t off) {
    return id - off - n_points;
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

typedef struct tree_builder_t {
    // vec of tree nodes
    tree_t *tree_nodes;

    // vec of root tree nodes
    // the root tree is the part of the tree which is cooperatively computed
    tree_t *tree_root_nodes;

    // mapping: idx in tree_root_nodes -> ID
    ssize_t *tree_root_node_ids;

    // number of root nodes
    ssize_t n_tree_root_nodes;

    // list of points
    double **points;

    // point coordinates
    double *points_values;

    // number of points
    ssize_t n_points;

    // number of active points
    ssize_t n_active_points;

    // number of local points
    ssize_t n_local_points;

    // list of inner products: preallocated to make sure this function is
    // no-alloc
    double *inner_products;

    // list of auxiliar inner products: preallocated to make sure this function
    // is no-alloc
    double *inner_products_aux;

    // id of the root of this subtree
    ssize_t *root_id;

    // id of this node
    ssize_t id;

    // index of the left-most point for this node
    ssize_t l;

    // index of the right-most point for this node
    ssize_t r;

    // id of the first point on this node
    ssize_t index_offset;

    // mpi proccess id in the communication group
    int proc_id;

    // number of mpi processes active in this communication group
    int n_procs;

    // communication group
    MPI_Comm comm;

    // number of available omp threads
    ssize_t omp_available;

    // depth of the local computation
    ssize_t omp_depth;
} tree_builder_t;

// Recurssion to build tree if subtree is all in memory
//
static void tree_build_single_aux(tree_builder_t b) {
    assert(b.r - b.l > 1, "1-sized trees are out of scope");

    tree_t *t = tree_index_to_ptr(b.tree_nodes, b.id - *b.root_id);
    // LOG("building tree node %zd: %p [%zd, %zd[ -> L = %zd, R = %zd", b.id,
    //(void*)t, b.l, b.r, tree_left_node_idx(b.id), tree_right_node_idx(b.id,
    // b.n_points));

    // double const begin = omp_get_wtime();
    divide_point_set(b.points, b.inner_products, b.inner_products_aux, b.l, b.r,
                     t->t_center, b.omp_available);

    if (b.proc_id % b.n_procs == 0) {
        t->t_radius = compute_radius(b.points, b.l, b.r, t->t_center);
    }

    // fprintf(stderr, "%zd %.12lf\n", b.omp_depth, omp_get_wtime() - begin);

    ssize_t m = (b.l + b.r) / 2;
    if (b.r - b.l == 2) {
        if (b.proc_id % b.n_procs == 0) {
            t->t_type = TREE_TYPE_BOTH_LEAF;
            t->t_left = tree_leaf_idx_to_id(b.l, b.n_points, b.index_offset);
            t->t_right =
                tree_leaf_idx_to_id(b.r - 1, b.n_points, b.index_offset);
        }
        return;
    }

    if (b.r - b.l == 3) {
        if (b.proc_id % b.n_procs == 0) {  // Just 1 of each set
            t->t_type = TREE_TYPE_LEFT_LEAF;
            t->t_left = tree_leaf_idx_to_id(b.l, b.n_points, b.index_offset);
            t->t_right = tree_right_node_idx(b.id, 3);

            b.id = t->t_right;
            b.l = m;
            b.omp_depth += 1;
            b.omp_available = 0;
            tree_build_single_aux(b);
        }
        return;
    }

    t->t_type = TREE_TYPE_INNER;
    t->t_left = tree_left_node_idx(b.id);
    t->t_right = tree_right_node_idx(b.id, b.r - b.l);

    if (b.n_procs == 1) {
        if (b.omp_available > 0) {  // Parallel
            b.omp_available -= (1 << (size_t)b.omp_depth);
            b.omp_depth += 1;
            tree_builder_t right_b = b;
#pragma omp parallel sections num_threads(2)
            {
#pragma omp section
                {
                    b.r = m;
                    b.id = t->t_left;
                    tree_build_single_aux(b);
                }

#pragma omp section
                {
                    right_b.l = m;
                    right_b.id = t->t_right;
                    tree_build_single_aux(right_b);
                }
            }
        } else {  // Serial
            tree_builder_t right_b = b;

            b.r = m;
            right_b.l = m;

            b.id = t->t_left;
            right_b.id = t->t_right;
            tree_build_single_aux(b);
            tree_build_single_aux(right_b);
        }

    } else if (b.proc_id % b.n_procs < b.n_procs / 2) {
        b.r = m;
        b.n_procs /= 2;
        b.id = t->t_left;
        b.omp_depth = 0;
        tree_build_single_aux(b);
    } else if (b.proc_id % b.n_procs >= b.n_procs / 2) {
        b.l = m;
        b.n_procs = (b.n_procs + 1) / 2;
        b.id = t->t_right;
        b.omp_depth = 0;
        tree_build_single_aux(b);
    }
}

// Compute the tree when tree fits in memory
//
static void tree_build_single(tree_t *tree_nodes, double **points,
                              double *inner_products, ssize_t n_points,
                              int proc_id, int n_procs) {
    omp_set_nested(1);
    ssize_t root_id = 0;
    tree_builder_t builder = {.tree_nodes = tree_nodes,
                              .tree_root_nodes = NULL,
                              .tree_root_node_ids = NULL,
                              .n_tree_root_nodes = 0,
                              .points = points,
                              .points_values = points[0],
                              .n_points = n_points,
                              .n_active_points = n_points,
                              .n_local_points = n_points,
                              .inner_products = inner_products,
                              .inner_products_aux = inner_products + n_points,
                              .id = 0,
                              .root_id = &root_id,
                              .l = 0,
                              .r = n_points,
                              .index_offset = n_points * proc_id,
                              .proc_id = proc_id,
                              .n_procs = n_procs,
                              .comm = MPI_COMM_WORLD,
                              .omp_available = omp_get_max_threads() - 1,
                              .omp_depth = 0};
    tree_build_single_aux(builder);
}

// Recurssion to build tree if subtree is distributed
//
static void tree_build_dist_aux(tree_builder_t b) {
    assert(b.r - b.l > 1, "1-sized trees are out of scope");

    tree_t *t = tree_index_to_ptr(b.tree_root_nodes, b.n_tree_root_nodes);
    // LOG("building tree node %zd: %p [%zd, %zd[ -> L = %zd, R = %zd", b.id,
    //(void*)t, b.l, b.r, tree_left_node_idx(b.id), tree_right_node_idx(b.id,
    // b.n_points));

    // double const begin = omp_get_wtime();

    dist_divide_point_set(b.points, b.points_values, b.inner_products,
                          b.inner_products_aux, b.n_local_points,
                          b.n_active_points, b.proc_id, b.n_procs, b.comm,
                          t->t_center);

    double radius = dist_compute_radius(b.points, b.n_local_points, b.proc_id,
                                        b.n_procs, b.comm, t->t_center);
    if (b.proc_id == 0) {
        t->t_radius = radius;
        b.tree_root_node_ids[b.n_tree_root_nodes] = b.id;
        b.n_tree_root_nodes += 1;
    }

    // fprintf(stderr, "%zd %.12lf\n", depth, omp_get_wtime() - begin);

    t->t_type = TREE_TYPE_INNER;
    t->t_left = tree_left_node_idx(b.id);
    t->t_right = tree_right_node_idx(b.id, b.n_active_points);

    int group = (b.proc_id < b.n_procs / 2) ? 0 : 1;
    MPI_Comm new_comm;
    MPI_Comm_split(b.comm, group, b.proc_id - group * (b.n_procs / 2),
                   &new_comm);

    b.comm = new_comm;
    MPI_Comm_rank(b.comm, &b.proc_id);
    MPI_Comm_size(b.comm, &b.n_procs);

    if (b.n_procs == 1) {
        if (group == 0) {
            b.id = t->t_left;
            b.n_active_points /= 2;

        } else {
            b.id = t->t_right;
            b.n_active_points = (b.n_active_points + 1) / 2;
        }
        *b.root_id = b.id;
        tree_build_single_aux(b);
    } else {  // still in distributed mode
        if (group == 0) {
            b.id = t->t_left;
            b.n_active_points /= 2;
        } else {
            b.id = t->t_right;
            b.n_active_points = (b.n_active_points + 1) / 2;
        }
        tree_build_dist_aux(b);
    }
}

// Compute the tree when tree is distributed
//
static void tree_build_dist(tree_t *tree_nodes, tree_t *tree_root_nodes,
                            ssize_t *tree_root_node_ids, double **points,
                            double *inner_products, ssize_t n_points,
                            ssize_t n_local_points, int proc_id, int n_procs,
                            ssize_t *root_id) {
    if (n_procs == 1) {
        tree_build_single(tree_nodes, points, inner_products, n_points, proc_id,
                          n_procs);
        return;
    }
    omp_set_nested(1);
    tree_builder_t builder = {
        .tree_nodes = tree_nodes,
        .tree_root_nodes = tree_root_nodes,
        .tree_root_node_ids = tree_root_node_ids,
        .n_tree_root_nodes = 0,
        .points = points,
        .points_values = points[0],
        .n_points = n_points,
        .n_active_points = n_points,
        .n_local_points = n_local_points,
        .inner_products = inner_products,
        .inner_products_aux = inner_products + n_local_points,
        .id = 0,
        .root_id = root_id,
        .l = 0,
        .r = n_local_points,
        .index_offset = n_points * proc_id,
        .proc_id = proc_id,
        .n_procs = n_procs,
        .comm = MPI_COMM_WORLD,
        .omp_available = omp_get_max_threads() - 1,
        .omp_depth = 0};
    tree_build_dist_aux(builder);
}

#ifndef PROFILE
static void tree_print_node(tree_t const *t, ssize_t id, double **points,
                            ssize_t n_points, ssize_t offset, int proc_id) {
    /*
    LOG("[%d] printing %zd {\n"
        "   type:   %d,\n"
        "   radius: %lf,\n"
        "   left:   %zd | %zd,\n"
        "   right:  %zd | %zd,\n"
        "}", proc_id, id, t->t_type, t->t_radius,
        t->t_left, (tree_has_left_leaf(t) ? tree_leaf_id_to_idx(t->t_left,
    n_points, offset) : -1), t->t_left, (tree_has_right_leaf(t) ?
    tree_leaf_id_to_idx(t->t_right, n_points, offset) : -1));
        */

    if (tree_has_left_leaf(t) != 0) {
        fprintf(stdout, "%zd -1 -1 %.6lf", t->t_left, 0.0);
        for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
            fprintf(
                stdout, " %lf",
                points[tree_leaf_id_to_idx(t->t_left, n_points, offset)][j]);
        }
        fputc('\n', stdout);
        fflush(stdout);
    }

    if (tree_has_right_leaf(t) != 0) {
        fprintf(stdout, "%zd -1 -1 %.6lf", t->t_right, 0.0);
        for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
            fprintf(
                stdout, " %lf",
                points[tree_leaf_id_to_idx(t->t_right, n_points, offset)][j]);
        }
        fputc('\n', stdout);
        fflush(stdout);
    }

    fprintf(stdout, "%zd %zd %zd %.6lf", id, t->t_left, t->t_right,
            t->t_radius);
    for (size_t j = 0; j < (size_t)N_DIMENSIONS; ++j) {
        fprintf(stdout, " %lf", t->t_center[j]);
    }
    fputc('\n', stdout);
    fflush(stdout);
}

static void tree_print(tree_t const *tree_nodes, tree_t const *tree_root_nodes,
                       ssize_t const *tree_root_node_ids, double **points,
                       ssize_t n_points, ssize_t n_local_points,
                       ssize_t sub_root_id, int proc_id, int n_procs) {
    ssize_t offset = n_points * proc_id;
    // nodes were calculated in a distributed manner
    if (tree_root_nodes != NULL) {

        for (int i = 0; i < n_procs - 1; ++i) {
            if (tree_root_node_ids[i] == -1) {
                continue;
            }

            tree_t const *t = tree_index_to_ptr(tree_root_nodes, i);
            if (t->t_radius == 0) {
                continue;
            }
            tree_print_node(t, tree_root_node_ids[i], points, n_points, offset,
                            proc_id);
        }
    }

    // nodes were calculated in memory
    for (ssize_t i = 0; i < n_local_points; ++i) {
        tree_t const *t = tree_index_to_ptr(tree_nodes, i);
        if (t->t_radius == 0) {
            continue;
        }
        tree_print_node(t, i + sub_root_id, points, n_points, offset, proc_id);
    }
}
#endif

#ifndef RANGE
#define RANGE 10
#endif  // RANGE

static ssize_t size_to_alloc(ssize_t n_points, int proc_id, int n_procs) {
    ssize_t div = n_points / n_procs;
    return (n_procs - (n_points % n_procs) <= proc_id) ? div + 1 : div;
}

static double **allocate(int *proc_id, int *n_procs, ssize_t n_points,
                         ssize_t *n_local_points, uint32_t seed,
                         tree_t **tree_nodes, tree_t **tree_root_nodes,
                         ssize_t **tree_root_node_ids, double **inner_products,
                         computation_mode_t *c_mode) {
    // We'll try to fit everything on the same machine.
    // If it fails, we only need n_points / n_procs
    *c_mode = CM_SINGLE_NODE;
    size_t sz_to_alloc = (size_t)n_points;

    // if active forces distributed mode
    if (DISTRIBUTED) {
        *c_mode = CM_DISTRIBUTED;
        sz_to_alloc = size_to_alloc(n_points, *proc_id, *n_procs);
    }

    srand(seed);

    double **pt_ptr = malloc(sizeof(double *) * sz_to_alloc);
    if (pt_ptr == NULL) {
        if (errno != ENOMEM) {
            KILL("Failed to allocate (and not due to lack of memory");
        }

        *c_mode = CM_DISTRIBUTED;
        sz_to_alloc = size_to_alloc(n_points, *proc_id, *n_procs);
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
        sz_to_alloc = size_to_alloc(n_points, *proc_id, *n_procs);
        pt_ptr = xrealloc(pt_ptr, sizeof(double *) * sz_to_alloc);
        pt_arr = xmalloc(sizeof(double) * (size_t)N_DIMENSIONS * sz_to_alloc);
    }

    *tree_nodes = calloc((size_t)(sz_to_alloc), tree_sizeof());
    if (*tree_nodes == NULL) {
        if (errno != ENOMEM) {
            KILL("Failed to allocate (and not due to lack of memory");
        } else if (*c_mode == CM_DISTRIBUTED) {
            KILL(
                "Even if we only allocate divided by the number of machines, "
                "we still OOM-out");
        }

        *c_mode = CM_DISTRIBUTED;
        sz_to_alloc = size_to_alloc(n_points, *proc_id, *n_procs);
        pt_ptr = xrealloc(pt_arr, sizeof(double *) * sz_to_alloc);
        pt_arr = xrealloc(pt_arr,
                          sizeof(double) * (size_t)N_DIMENSIONS * sz_to_alloc);
        *tree_nodes = xcalloc((size_t)(sz_to_alloc), tree_sizeof());
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
        sz_to_alloc = size_to_alloc(n_points, *proc_id, *n_procs);
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
     *   on a set of distributed nodes, which may not exist)
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

    if (*c_mode == CM_DISTRIBUTED) {
        // This is a small array: if we fail we are in bad shape
        *tree_root_nodes = xcalloc((size_t)*n_procs - 1, tree_sizeof());
        *tree_root_node_ids = xmalloc(((size_t)*n_procs - 1) * sizeof(ssize_t));

        for (ssize_t idx = 0; idx < *n_procs - 1; ++idx) {
            (*tree_root_node_ids)[idx] = -1;
        }
    }

    *n_local_points = (size_t)sz_to_alloc;
    if (*c_mode == CM_SINGLE_NODE) {
// As discussed, the number of inner nodes is
// at most the number of leaves of the tree.
//
#pragma omp parallel sections
        {
#pragma omp section
            {
                for (ssize_t i = 0; i < n_points; i++) {
                    for (ssize_t j = 0; j < N_DIMENSIONS; j++) {
                        pt_arr[i * (N_DIMENSIONS) + j] =
                            RANGE * ((double)rand()) / RAND_MAX;
                    }
                }
            }
#pragma omp section
            {
                for (ssize_t i = 0; i < n_points; i++) {
                    pt_ptr[i] = &pt_arr[i * (N_DIMENSIONS)];
                }
            }
        }
    } else if (*c_mode == CM_DISTRIBUTED) {
        int generating_id = 0;
        ssize_t to_generate = size_to_alloc(n_points, generating_id, *n_procs);
        for (ssize_t i = 0, idx = 0; i < n_points; ++i) {
            if (generating_id == *proc_id) {
                break;
            }
            for (ssize_t j = 0; j < N_DIMENSIONS; ++j) {
                double r = rand();
            }

            to_generate -= 1;
            if (to_generate == 0) {
                generating_id += 1;
                to_generate = size_to_alloc(n_points, generating_id, *n_procs);
            }
        }

#pragma omp parallel sections
        {
#pragma omp section
            {
                for (ssize_t i = 0; i < *n_local_points; i++) {
                    for (ssize_t j = 0; j < N_DIMENSIONS; j++) {
                        pt_arr[i * (N_DIMENSIONS) + j] =
                            RANGE * ((double)rand()) / RAND_MAX;
                    }
                }
            }
#pragma omp section
            {
                for (ssize_t i = 0; i < *n_local_points; i++) {
                    pt_ptr[i] = &pt_arr[i * (N_DIMENSIONS)];
                }
            }
        }
    }

    return pt_ptr;
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
    tree_t *tree_root_nodes = NULL;
    ssize_t *tree_root_node_ids = NULL;
    double *inner_products = NULL;
    computation_mode_t c_mode = CM_SINGLE_NODE;
    int old_id = proc_id;
    double **points = allocate(&proc_id, &n_procs, n_points, &n_local_points,
                               seed, &tree_nodes, &tree_root_nodes,
                               &tree_root_node_ids, &inner_products, &c_mode);
    double const *point_values = points[0];

    if (proc_id == 0) {
        fprintf(stdout, "%zd %zd\n", N_DIMENSIONS, 2 * n_points - 1);
        fflush(stdout);
    }

    ssize_t root_id = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    switch (c_mode) {
        case CM_SINGLE_NODE:
            tree_build_single(tree_nodes, points, inner_products, n_points,
                              proc_id, n_procs);
            break;
        case CM_DISTRIBUTED:
            tree_build_dist(tree_nodes, tree_root_nodes, tree_root_node_ids,
                            points, inner_products, n_points, n_local_points,
                            proc_id, n_procs, &root_id);
            break;
        case CM_PASSIVE:
            break;
        default:
            __builtin_unreachable();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_id == 0) {
        fprintf(stderr, "%.1lf\n", MPI_Wtime() - begin);
        fflush(stderr);
#ifndef PROFILE
        tree_print(tree_nodes, tree_root_nodes, tree_root_node_ids, points,
                   n_points, n_local_points, root_id, proc_id, n_procs);
        fflush(stdout);
#endif
    }

#ifndef PROFILE
    for (int pid = 1; pid < n_procs; ++pid) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (proc_id == pid) {
            tree_print(tree_nodes, tree_root_nodes, tree_root_node_ids, points,
                       n_points, n_local_points, root_id, proc_id, n_procs);
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
