#include <omp.h>
#include "main.h"

int main(int argc, char *argv[]) {
    return strategy_main(argc, argv, most_distant_approx_parallel);
}
