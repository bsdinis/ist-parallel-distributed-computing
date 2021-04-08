/***
 * util functions for memory management
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * checked malloc
 */
void *priv_xmalloc__(char const *file, int lineno, size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL && size != 0) {
        fprintf(stderr, "[ERROR] %s:%d | xmalloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return ptr;
}

/**
 * checked realloc
 */
void *priv_xrealloc__(char const *file, int lineno, void *ptr, size_t size) {
    void *new_ptr = realloc(ptr, size);
    if (size != 0 && new_ptr == NULL && ptr != NULL) {
        fprintf(stderr, "[ERROR] %s:%d | xrealloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return new_ptr;
}

/**
 * checked calloc
 */
void *priv_xcalloc__(char const *file, int lineno, size_t nmemb, size_t size) {
    void *ptr = calloc(nmemb, size);
    if (ptr == NULL && size != 0) {
        fprintf(stderr, "[ERROR] %s:%d | xcalloc failed: %s\n", file, lineno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    return ptr;
}
