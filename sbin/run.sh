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
        ./ballAlg $(cat ${dir}/alg) > output.tree
        break
    done
done
