printf '\n add_cpu.cu (single precision) \n'
nvcc -O3 -arch=sm_35 add_cpu.cu -o add_cpu_single
./add_cpu_single

printf '\n add_cpu.cu (double precision) \n'
nvcc -O3 -arch=sm_35 -DUSE_DP add_cpu.cu -o add_cpu_double
./add_cpu_double

printf '\n add_gpu.cu (single precision) \n'
nvcc -O3 -arch=sm_35 add_gpu.cu -o add_gpu_single
./add_gpu_single

printf '\n add_gpu.cu (double precision) \n'
nvcc -O3 -arch=sm_35 -DUSE_DP add_gpu.cu -o add_gpu_double
./add_gpu_double

printf '\n add_gpu_memcpy.cu (single precision) \n'
nvcc -O3 -arch=sm_35 add_gpu_memcpy.cu -o add_gpu_memcpy_single
./add_gpu_memcpy_single

printf '\n add_gpu_memcpy.cu (double precision) \n'
nvcc -O3 -arch=sm_35 -DUSE_DP add_gpu_memcpy.cu -o add_gpu_memcpy_double
./add_gpu_memcpy_double

printf '\n arithmetic_cpu.cu (single precision) \n'
nvcc -O3 -arch=sm_35 arithmetic_cpu.cu -o arithmetic_cpu_single
./arithmetic_cpu_single

printf '\n arithmetic_cpu.cu (double precision) \n'
nvcc -O3 -arch=sm_35 -DUSE_DP arithmetic_cpu.cu -o arithmetic_cpu_double
./arithmetic_cpu_double

printf '\n arithmetic_gpu.cu (single precision) \n'
nvcc -O3 -arch=sm_35 arithmetic_gpu.cu -o arithmetic_gpu_single
./arithmetic_gpu_single 1000000

printf '\n arithmetic_gpu.cu (double precision) \n'
nvcc -O3 -arch=sm_35 -DUSE_DP arithmetic_gpu.cu -o arithmetic_gpu_double
./arithmetic_gpu_double 1000000




