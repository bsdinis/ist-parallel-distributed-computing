#include "main.h"
#include <omp.h>

int main(int argc, char *argv[]) {
    omp_set_num_threads(1);
    return strategy_main(argc, argv, most_distant_approx_serial);
}
