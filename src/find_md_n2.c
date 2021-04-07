// Returns:
//    a and b by reference (as indeces)
//    l and r define a range to search (r is exclusive)
//    distance squared
static double find_two_most_distant(double const **points, ssize_t l, ssize_t r,
                                    ssize_t *a, ssize_t *b) {
    ssize_t max_a = 0;
    ssize_t max_b = 0;
    double long_dist = 0;

    // O(n**2)
    for (ssize_t i = l; i < r; i++) {
        for (ssize_t j = i + 1; j < r; j++) {
            double aux_long_dist =
                distance_squared(points[i], points[j]);
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