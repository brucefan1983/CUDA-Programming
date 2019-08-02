#!/bin/sh

printf '\nnvcc -arch=sm_35 global.cu\n'
nvcc -arch=sm_35 global.cu
printf './a.out\n'
./a.out
