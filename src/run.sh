#!/bin/sh

printf '\n===================================================\n'
printf 'cd 02*'
printf '\n===================================================\n'
cd 02*

printf '\ng++ hello.cpp\n'
g++ hello.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 hello1.cu\n'
nvcc -arch=sm_35 hello1.cu
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 hello2.cu\n'
nvcc -arch=sm_35 hello2.cu
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 hello3.cu\n'
nvcc -arch=sm_35 hello3.cu
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 hello4.cu\n'
nvcc -arch=sm_35 hello4.cu
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 hello5.cu\n'
nvcc -arch=sm_35 hello5.cu
printf '\n./a.out\n'
./a.out

printf '\n===================================================\n'
printf 'cd ../03*'
printf '\n===================================================\n'
cd ../03*

printf '\ng++ add.cpp\n'
g++ add.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 add1.cu\n'
nvcc -arch=sm_35 add1.cu
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 add2wrong.cu\n'
nvcc -arch=sm_35 add2wrong.cu
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 add3if.cu\n'
nvcc -arch=sm_35 add3if.cu
printf '\n./a.out\n'
./a.out

printf '\n===================================================\n'
printf 'cd ../04*'
printf '\n===================================================\n'
cd ../04*

printf '\nnvcc -arch=sm_35 add4check_api.cu\n'
nvcc add4check_api.cu
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 add5check_kernel.cu\n'
nvcc add5check_kernel.cu
printf '\n./a.out\n'
./a.out

printf '\n===================================================\n'
printf 'cd ../05*'
printf '\n===================================================\n'
cd ../05*

printf '\ng++ -O3 add.cpp\n'
g++ -O3 add.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 add.cu\n'
nvcc -arch=sm_35 add.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 1\n'
nvprof --unified-memory-profiling off ./a.out 1
printf '\nnvprof --unified-memory-profiling off ./a.out 1000\n'
nvprof --unified-memory-profiling off ./a.out 1000

printf '\nnvcc -arch=sm_35 -DUSE_DP add.cu\n'
nvcc -arch=sm_35 -DUSE_DP add.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 1\n'
nvprof --unified-memory-profiling off ./a.out 1

printf '\ng++ -O3 arithmetic.cpp\n'
g++ -O3 arithmetic.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_35 arithmetic.cu\n'
nvcc -arch=sm_35 arithmetic.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 100000000 128\n'
nvprof --unified-memory-profiling off ./a.out 100000000 128
printf '\nnvprof --unified-memory-profiling off ./a.out 10000000 128\n'
nvprof --unified-memory-profiling off ./a.out 10000000 128
printf '\nnvprof --unified-memory-profiling off ./a.out 1000000 128\n'
nvprof --unified-memory-profiling off ./a.out 1000000 128
printf '\nnvprof --unified-memory-profiling off ./a.out 100000 128\n'
nvprof --unified-memory-profiling off ./a.out 100000 128
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 128\n'
nvprof --unified-memory-profiling off ./a.out 10000 128
printf '\nnvprof --unified-memory-profiling off ./a.out 1000 128\n'
nvprof --unified-memory-profiling off ./a.out 1000 128
printf '\nnvprof --unified-memory-profiling off ./a.out 100000000 32\n'
nvprof --unified-memory-profiling off ./a.out 100000000 32
printf '\nnvprof --unified-memory-profiling off ./a.out 100000000 64\n'
nvprof --unified-memory-profiling off ./a.out 100000000 64
printf '\nnvprof --unified-memory-profiling off ./a.out 100000000 256\n'
nvprof --unified-memory-profiling off ./a.out 100000000 256
printf '\nnvprof --unified-memory-profiling off ./a.out 100000000 512\n'
nvprof --unified-memory-profiling off ./a.out 100000000 512
printf '\nnvprof --unified-memory-profiling off ./a.out 100000000 1024\n'
nvprof --unified-memory-profiling off ./a.out 100000000 1024

printf '\n===================================================\n'
printf 'cd ../07*'
printf '\n===================================================\n'
cd ../07*

printf '\nnvcc -arch=sm_35 add_unified.cu\n'
nvcc -arch=sm_35 add_unified.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out

printf '\n===================================================\n'
printf 'cd ../08*'
printf '\n===================================================\n'
cd ../08*

printf '\nnvcc -arch=sm_35 copy.cu\n'
nvcc -arch=sm_35 copy.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_35 transpose1global_coalesced_read.cu\n'
nvcc -arch=sm_35 transpose1global_coalesced_read.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_35 transpose2global_coalesced_write.cu\n'
nvcc -arch=sm_35 transpose2global_coalesced_write.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_35 transpose3global_ldg.cu\n'
nvcc -arch=sm_35 transpose3global_ldg.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_35 transpose4shared_with_confilict.cu\n'
nvcc -arch=sm_35 transpose4shared_with_conflict.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_35 transpose5shared_without_conflict.cu\n'
nvcc -arch=sm_35 transpose5shared_without_conflict.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16


