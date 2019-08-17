#!/bin/sh

printf '\ng++ sum.cpp\n'
g++ sum.cpp
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum1.cu\n'
nvcc -arch=sm_35 sum1.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum2wrong.cu\n'
nvcc -arch=sm_35 sum2wrong.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum3if.cu\n'
nvcc -arch=sm_35 sum3if.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum4check_api.cu\n'
nvcc -arch=sm_35 sum4check_api.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum5check_kernel.cu\n'
nvcc -arch=sm_35 sum5check_kernel.cu
echo './a.out'
./a.out




