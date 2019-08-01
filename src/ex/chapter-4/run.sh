#!/bin/sh

printf '\nnvcc -arch=sm_35 copy.cu\n'
nvcc -arch=sm_35 copy.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out
