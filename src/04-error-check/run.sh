printf '\n add_check_api.cu \n'
nvcc add_check_api.cu -o add_check_api
./add_check_api

printf '\n add_check_kernel.cu \n'
nvcc add_check_kernel.cu -o add_check_kernel
./add_check_kernel

printf '\n add_memcheck.cu \n'
nvcc add_memcheck.cu -o add_memcheck
cuda-memcheck ./add_memcheck

