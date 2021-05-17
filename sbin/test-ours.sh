#!/usr/bin/env zsh

for i in 1 2 4 5 8;
do
    for rep in 1 2 3;
    do
        echo "ours $i $rep"
        OMP_NUM_THREADS=$i ./ballAlg-omp 4 20000000 0 2> approach_ours_${i}_${rep}.time
    done
done

