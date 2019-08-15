#!/bin/sh

make clean
make

for i in $(seq 5 5 30)
do
    echo "$i"
    nvprof --unified-memory-profiling off ./ljmd $i
done

