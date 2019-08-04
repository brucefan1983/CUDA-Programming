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

printf '\nnvcc -arch=sm_35 sum4check.cu\n'
nvcc -arch=sm_35 sum4check.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum5memcheck.cu\n'
nvcc -arch=sm_35 sum5memcheck.cu
echo 'cuda-memcheck ./a.out'
cuda-memcheck ./a.out



