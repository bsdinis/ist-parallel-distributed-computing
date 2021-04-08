#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

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

void *priv_xmalloc__(char const *file, int lineno, size_t size);
void *priv_xrealloc__(char const *file, int lineno, void *ptr, size_t size);
void *priv_xcalloc__(char const *file, int lineno, size_t nmemb, size_t size);
