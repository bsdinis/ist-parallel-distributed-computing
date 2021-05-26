#!/usr/bin/env zsh

for rep in 1 2 3 4 5;
do
    echo "ours $i $rep"
    OMP_NUM_THREADS=1 ./ballAlg-omp $(cat tests/0007_l/alg) 2> levels_${rep}.time
done
