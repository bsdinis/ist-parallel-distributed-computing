#!/usr/bin/env zsh

total=0
success=0

RED="\033[0;31m"
GREEN="\033[0;32m"
RESET="\033[0m"

if [[ $# -eq 0 ]]
then
    for dir in tests/*;
    do
        total=$(echo $total + 1 | bc)
        echo -n "$(basename $dir)... "
        OMP_N_THREADS=1 mpirun -n 2 ./ballAlg-mpi $(cat ${dir}/alg) 2>${dir}/time | ./ballQuery_pipe $(cat ${dir}/query) > ${dir}/output
        diff -b ${dir}/output ${dir}/expected > /dev/null 2> /dev/null;
        if [ $? -eq 0 ]
        then
            success=$(echo $success + 1 | bc)
            echo "${GREEN}OK${RESET}";
        else
            echo "${RED}NOK${RESET}";
        fi
    done
else
    for t in $@;
    do
        test_n=$(printf "%04d" $(echo $t | cut -d_ -f 1))

        for dir in tests/${test_n}*;
        do
            echo -n "$(basename $dir)... "
            OMP_N_THREADS=1 mpirun -n 2 ./ballAlg-mpi $(cat ${dir}/alg) 2>${dir}/time | ./ballQuery_pipe $(cat ${dir}/query) > ${dir}/output
            diff -b ${dir}/output ${dir}/expected;
            if [ $? -eq 0 ]
            then
                echo "${GREEN}OK${RESET}";
            else
                echo "${RED}NOK${RESET}";
            fi
        done
    done
fi
