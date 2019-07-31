#!/bin/sh

echo 'g++ hello.cpp'
g++ hello.cpp
echo './a.out'
./a.out

echo 'nvcc -arch=sm_35 hello1.cu'
nvcc -arch=sm_35 hello1.cu
echo './a.out'
./a.out

echo 'nvcc -arch=sm_35 hello2.cu'
nvcc -arch=sm_35 hello2.cu
echo './a.out'
./a.out

echo 'nvcc -arch=sm_35 hello3.cu'
nvcc -arch=sm_35 hello3.cu
echo './a.out'
./a.out

echo 'nvcc -arch=sm_35 hello4.cu'
nvcc -arch=sm_35 hello4.cu
echo './a.out'
./a.out

echo 'nvcc -arch=sm_35 hello5.cu'
nvcc -arch=sm_35 hello5.cu
echo './a.out'
./a.out
