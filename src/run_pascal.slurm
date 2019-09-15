#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 30:00
#SBATCH -J gpumd
#SBATCH -o gpumd.out.%j
#SBATCH -e gpumd.out.%j
#SBATCH --gres=gpu:p100:1
#SBATCH

module purge
module load gcc cuda
module list


printf '\n===================================================\n'
printf 'cd 05*'
printf '\n===================================================\n'
cd 05*

printf '\ng++ -O3 add.cpp\n'
g++ -O3 add.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_60 add.cu\n'
nvcc -arch=sm_60 add.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 1\n'
nvprof --unified-memory-profiling off ./a.out 1
printf '\nnvprof --unified-memory-profiling off ./a.out 1000\n'
nvprof --unified-memory-profiling off ./a.out 1000

printf '\nnvcc -arch=sm_60 -DUSE_DP add.cu\n'
nvcc -arch=sm_60 -DUSE_DP add.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 1\n'
nvprof --unified-memory-profiling off ./a.out 1

printf '\ng++ -O3 arithmetic.cpp\n'
g++ -O3 arithmetic.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_60 arithmetic.cu\n'
nvcc -arch=sm_60 arithmetic.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 100000000\n'
nvprof --unified-memory-profiling off ./a.out 100000000
printf '\nnvprof --unified-memory-profiling off ./a.out 10000000\n'
nvprof --unified-memory-profiling off ./a.out 10000000
printf '\nnvprof --unified-memory-profiling off ./a.out 1000000\n'
nvprof --unified-memory-profiling off ./a.out 1000000
printf '\nnvprof --unified-memory-profiling off ./a.out 100000\n'
nvprof --unified-memory-profiling off ./a.out 100000
printf '\nnvprof --unified-memory-profiling off ./a.out 10000\n'
nvprof --unified-memory-profiling off ./a.out 10000
printf '\nnvprof --unified-memory-profiling off ./a.out 1000\n'
nvprof --unified-memory-profiling off ./a.out 1000

printf '\n===================================================\n'
printf 'cd ../07*'
printf '\n===================================================\n'
cd ../07*

printf '\nnvcc -arch=sm_60 copy.cu\n'
nvcc -arch=sm_60 copy.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose1global_coalesced_read.cu\n'
nvcc -arch=sm_60 transpose1global_coalesced_read.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose2global_coalesced_write.cu\n'
nvcc -arch=sm_60 transpose2global_coalesced_write.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose3global_ldg.cu\n'
nvcc -arch=sm_60 transpose3global_ldg.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose4shared_with_confilict.cu\n'
nvcc -arch=sm_60 transpose4shared_with_conflict.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose5shared_without_conflict.cu\n'
nvcc -arch=sm_60 transpose5shared_without_conflict.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\n===================================================\n'
printf 'cd ../08*'
printf '\n===================================================\n'
cd ../08*

printf '\ng++ -O3 reduce1.cpp\n'
g++ -O3 reduce1.cpp
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_60 reduce2one_kernel.cu\n'
nvcc -arch=sm_60 reduce2one_kernel.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out

printf '\nnvcc -arch=sm_60 reduce3two_kernels.cu\n'
nvcc -arch=sm_60 reduce3two_kernels.cu
printf '\nnvprof --unified-memory-profiling off ./a.out\n'
nvprof --unified-memory-profiling off ./a.out

printf '\nnvcc -arch=sm_60 reduce4more_parallelism.cu\n'
nvcc -arch=sm_60 reduce4more_parallelism.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10\n'
nvprof --unified-memory-profiling off ./a.out 10

printf '\n===================================================\n'
printf 'cd ../09*'
printf '\n===================================================\n'
cd ../09*

printf '\nnvcc -arch=sm_60 reduce5atomic.cu\n'
nvcc -arch=sm_60 reduce5atomic.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10\n'
nvprof --unified-memory-profiling off ./a.out 10

printf '\n===================================================\n'
printf 'cd ../10*'
printf '\n===================================================\n'
cd ../10*

printf '\nnvcc -arch=sm_60 reduce7syncwarp.cu\n'
nvcc -arch=sm_60 reduce7syncwarp.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10\n'
nvprof --unified-memory-profiling off ./a.out 10

printf '\nnvcc -arch=sm_60 reduce8shfl.cu\n'
nvcc -arch=sm_60 reduce8shfl.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10\n'
nvprof --unified-memory-profiling off ./a.out 10

printf '\nnvcc -arch=sm_60 reduce9cp.cu\n'
nvcc -arch=sm_60 reduce9cp.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10\n'
nvprof --unified-memory-profiling off ./a.out 10

printf '\n===================================================\n'
printf 'cd ../11*'
printf '\n===================================================\n'
cd ../11*

printf '\nnvcc -arch=sm_60 host_kernel.cu\n'
nvcc -arch=sm_60 host_kernel.cu
printf '\n./a.out\n'
./a.out

printf '\nnvcc -arch=sm_60 kernel_kernel.cu\n'
nvcc -arch=sm_60 kernel_kernel.cu
printf '\n./a.out>t.txt\n'
./a.out>t.txt

printf '\nnvcc -arch=sm_60 kernel_transfer.cu\n'
nvcc -arch=sm_60 kernel_transfer.cu
printf '\n./a.out>t2.txt\n'
./a.out>t2.txt

printf '\n===================================================\n'
printf 'cd ../16*'
printf '\n===================================================\n'
cd ../16*

printf '\nmake -j6\n'
make -j6
printf '\n./ljmd 8 10000\n'
./ljmd 8 10000

printf '\n===================================================\n'
printf 'cd ../17*'
printf '\n===================================================\n'
cd ../17*

printf '\nmake -j6\n'
make -j6
printf '\n./ljmd 40 10000\n'
./ljmd 40 10000
printf '\nnvprof --unified-memory-profiling off ./ljmd 40 10000\n'
nvprof --unified-memory-profiling off ./ljmd 40 10000

printf '\n===================================================\n'
printf 'cd ../18*'
printf '\n===================================================\n'
cd ../18*

printf '\nmake -j6\n'
make -j6
printf '\n./ljmd 40 10000\n'
./ljmd 40 10000
printf '\nnvprof --unified-memory-profiling off ./ljmd 40 10000\n'
nvprof --unified-memory-profiling off ./ljmd 40 10000

printf '\n===================================================\n'
printf 'cd ../19*'
printf '\n===================================================\n'
cd ../19*

printf '\nmake -j6\n'
make -j6
printf '\n./ljmd 40 10000\n'
./ljmd 40 10000
printf '\nnvprof --unified-memory-profiling off ./ljmd 40 10000\n'
nvprof --unified-memory-profiling off ./ljmd 40 10000

printf '\nmake clean\n'
make clean
printf '\nmake -j6 -f makefile.ldg\n'
make -j6 -f makefile.ldg
printf '\n./ljmd 40 10000\n'
./ljmd 40 10000
printf '\nnvprof --unified-memory-profiling off ./ljmd 40 10000\n'
nvprof --unified-memory-profiling off ./ljmd 40 10000

printf '\nmake clean\n'
make clean
printf '\nmake -j6 -f makefile.fast_math\n'
make -j6 -f makefile.fast_math
printf '\n./ljmd 40 10000\n'
./ljmd 40 10000
printf '\nnvprof --unified-memory-profiling off ./ljmd 40 10000\n'
nvprof --unified-memory-profiling off ./ljmd 40 10000

printf '\n===================================================\n'
printf 'cd ../20*'
printf '\n===================================================\n'
cd ../20*

printf '\nmake -j6 -f makefile.pascal\n'
make -j6 -f makefile.pascal
printf '\n./ljmd 40 10000\n'
./ljmd 40 10000
printf '\nnvprof --unified-memory-profiling off ./ljmd 40 10000\n'
nvprof --unified-memory-profiling off ./ljmd 40 10000

