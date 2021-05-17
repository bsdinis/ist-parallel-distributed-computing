#!/usr/bin/env zsh

make clean
make -j all PROFILE=1

for dir in 0003_m 0004_m 0005_l 0006_l 0007_l;
do
    for i in 1 2 4 5 8;
    do
        for rep in 1 2 3;
        do
            echo "ours $i $rep"
            OMP_NUM_THREADS=$i ./ballAlg-omp $(cat tests/$dir/alg) 2> benchmark_${dir}_${i}_${rep}.time
        done
    done
done
