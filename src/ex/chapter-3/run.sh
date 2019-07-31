#!/bin/sh

printf '\nnvcc -arch=sm_35 sum6new.cu\n'
nvcc -arch=sm_35 sum6new.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum7segmentation.cu\n'
nvcc -arch=sm_35 sum7segmentation.cu
echo './a.out'
./a.out

printf '\nnvcc -arch=sm_35 sum8slow.cu\n'
nvcc -arch=sm_35 sum8slow.cu
echo './a.out'
./a.out



