#!/usr/bin/env bash

echo "hello"
echo $@
for t in $@;
do
    test_n=$(printf "%04d" $(echo $t | cut -d_ -f 1))
    echo $test_n;

    for dir in tests/${test_n}*;
    do
        echo -n "$(basename $dir)... "
        cd perf;
        perf record -F 99 --call-graph dwarf ../ballAlg $(cat ${dir}/alg)
        perf script | inferno-collapse-perf > stacks.folded
        cat stacks.folded | inferno-flamegraph > flamegraph.svg
        break
    done
done
