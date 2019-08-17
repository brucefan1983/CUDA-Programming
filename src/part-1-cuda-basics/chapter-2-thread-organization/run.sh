#!/bin/sh

printf '\ng++ hello.cpp\n'
g++ hello.cpp
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 hello1.cu\n'
nvcc hello1.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 hello2.cu\n'
nvcc hello2.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 hello3.cu\n'
nvcc hello3.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 hello4.cu\n'
nvcc hello4.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 hello5.cu\n'
nvcc hello5.cu
echo './a.out'
./a.out
