#!/bin/sh

printf '\n===================================================\n'
printf 'cd 07*'
printf '\n===================================================\n'
cd 07*

printf '\nnvcc -arch=sm_60 copy.cu\n'
nvcc -arch=sm_50 copy.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose1global_coalesced_read.cu\n'
nvcc -arch=sm_50 transpose1global_coalesced_read.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose2global_coalesced_write.cu\n'
nvcc -arch=sm_50 transpose2global_coalesced_write.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose3global_ldg.cu\n'
nvcc -arch=sm_50 transpose3global_ldg.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose4shared_with_confilict.cu\n'
nvcc -arch=sm_50 transpose4shared_with_conflict.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16
printf '\nnvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request ./a.out 10000 16 16\n'
nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request ./a.out 10000 16 16

printf '\nnvcc -arch=sm_60 transpose5shared_without_conflict.cu\n'
nvcc -arch=sm_50 transpose5shared_without_conflict.cu
printf '\nnvprof --unified-memory-profiling off ./a.out 10000 16 16\n'
nvprof --unified-memory-profiling off ./a.out 10000 16 16
printf '\nnvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request ./a.out 10000 16 16\n'
nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request ./a.out 10000 16 16

