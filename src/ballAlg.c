
// D**2 = Sum {v=1 to N_DIMENSIONS} (pt1_x - pt2_x)**2
static inline double distance_squared(double const *pt_1, double const *pt_2) {
    double d_s = 0;
    for (ssize_t i = 0; i < N_DIMENSIONS; i++) {
        double aux = (pt_1[i] - pt_2[i]);
        d_s += aux * aux;
    }
    return d_s;
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
                                double *products, double median) {
    ssize_t i = 0;
    ssize_t j = r - l - 1;
    ssize_t k = ( r - l ) / 2;
    while (i < j) {
        while (i < j && products[i] < median) {
            i++;
        }
        while (i < j && products[j] > median) {
            j--;
        }
        if (i < j) {
            if(products[i] == median){ //i and j will swap
                k = j;
            }
            else if(products[j] == median){
                k = i;
            }
            swap_ptr((void **)&points[l + i], (void **)&points[l + j]);
            i++;
            j--;
        }
    }
    ssize_t m = (r - l) / 2 ;
    swap_ptr((void **)&points[l + k], (void **)&points[l + m]); // ensure medium is on the right set
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

    double *products = xmalloc((size_t)(r - l)* 2 * sizeof(double));
    double *products_aux = products + r - l;
    for (ssize_t i = 0; i < r - l; ++i) {
        products[i] = diff_inner_product(points[l + i], points[a], b_minus_a);
        products_aux[i] = products[i];
    }

    // O(n)
    double median = find_median(products, (r - l));

    // O(n)
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
                              ssize_t idx, ssize_t l, ssize_t r) {
    assert(r - l > 1, "1-sized trees are out of scope");

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
                          ssize_t n_points) {
    return tree_build_aux(tree_nodes, points, 0, 0, n_points);
}

#ifndef RANGE
#define RANGE 10
#endif

#ifndef ssize_t
#define ssize_t __ssize_t
#endif

 ssize_t N_DIMENSIONS = 0;

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

