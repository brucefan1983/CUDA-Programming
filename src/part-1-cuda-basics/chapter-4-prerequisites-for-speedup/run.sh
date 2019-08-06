#!/bin/sh

printf '\ng++ -O3 sum.cpp\n'
g++ -O3 sum.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 sum1.cu\n'
nvcc -arch=sm_35 sum1.cu
printf '\n./a.out\n'
./a.out

printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out

printf '\nnvcc -arch=sm_35 sum2.cu\n'
nvcc -arch=sm_35 sum2.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out

printf '\nnvcc -arch=sm_35 copy.cu\n'
nvcc -arch=sm_35 copy.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out

printf '\ng++ -O3 pow.cpp\n'
g++ -O3 pow.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 pow.cu\n'
nvcc -arch=sm_35 pow.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out
