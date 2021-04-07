extern double N_DIMENSIONS;

// Returns:
//    a and b by reference (as indeces)
//    l and r define a range to search (r is exclusive)
//    distance squared
static double find_two_most_distant(double const **points, ssize_t l, ssize_t r,
                                    ssize_t *a, ssize_t *b) {
    double * centroid = xcalloc(N_DIMENSIONS, sizeof(double));

    for (ssize_t i = l; i < r; i++) {
        for (ssize_t d = 0; d < N_DIMENSIONS; d++) {
            centroid[d] += points[i][d];
        }
    }
    for (ssize_t d = 0; d < N_DIMENSIONS; d++) {
        centroid[d] /= N_DIMENSIONS;
    }

    double dist_l_a = 0;
    for (ssize_t i = l + 1; i < r; ++i) {
        double dist = distance_squared(centroid, points[i]);
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

    free(centroid);

    return dist_a_b;
}