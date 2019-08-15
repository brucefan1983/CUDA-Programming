#!/bin/sh

make clean
make

for i in $(seq 5 5 30)
do
    echo "$i"
    ./ljmd $i
done







