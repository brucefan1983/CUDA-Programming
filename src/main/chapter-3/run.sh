#!/bin/sh

printf '\ng++ sum.cpp\n'
g++ sum.cpp
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum1.cu\n'
nvcc -arch=sm_35 sum1.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum2.cu\n'
nvcc -arch=sm_35 sum1.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum3wrong.cu\n'
nvcc -arch=sm_35 sum3wrong.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum4wrong.cu\n'
nvcc -arch=sm_35 sum4wrong.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum5slow.cu\n'
nvcc -arch=sm_35 sum5slow.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum6.cu\n'
nvcc -arch=sm_35 sum6.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum7check.cu\n'
nvcc -arch=sm_35 sum7check.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum8memcheck.cu\n'
nvcc -arch=sm_35 sum8memcheck.cu
echo './a.out'
cuda-memcheck ./a.out



