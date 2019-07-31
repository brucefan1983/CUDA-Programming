#!/bin/sh

printf "\nnvcc -arch=sm_35 hello6.cu\n"
nvcc -arch=sm_35 hello6.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 hello7.cu\n'
nvcc -arch=sm_35 hello7.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 hello8.cu\n'
nvcc -arch=sm_35 hello8.cu
echo './a.out'
./a.out


