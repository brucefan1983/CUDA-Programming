#include "error.cuh"
#include <stdio.h>
#include <time.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 1000;
void mem_alloc_and_copy(const int mem_size);
void mem_alloc_only(const int mem_size);
void get_time(const int mem_size);

int main(void)
{
    printf("mem_size (KB) t_alloc (ms) t_copy (ms)\n");
    mem_alloc_and_copy(10000);
    for (int n = 15; n < 23; ++n)
    {
        get_time((1 << n) * sizeof(real));
    }
    return 0;
}

void mem_alloc_and_copy(const int mem_size)
{ 
    real *y;
    CHECK(cudaMalloc(&y, mem_size))
    real *h_y = (real *)malloc(mem_size);
    CHECK(cudaMemcpy(h_y, y, mem_size, cudaMemcpyDeviceToHost))
    free(h_y);
    CHECK(cudaFree(y))
}

void mem_alloc_only(const int mem_size)
{ 
    real *y;
    CHECK(cudaMalloc(&y, mem_size))
    CHECK(cudaFree(y))
}

void get_time(const int mem_size)
{
    clock_t t1 = clock();
    for (int n = 0; n < NUM_REPEATS; n++)
    {
        mem_alloc_only(mem_size);
    }
    clock_t t2 = clock();
    real t_alloc = (t2 - t1) / real(CLOCKS_PER_SEC);

    t1 = clock();
    for (int n = 0; n < NUM_REPEATS; n++)
    {
        mem_alloc_and_copy(mem_size);
    }
    t2 = clock();
    real t_copy = (t2 - t1) / real(CLOCKS_PER_SEC) - t_alloc;

    printf("%13d%13g%12g\n", mem_size/1024, t_alloc, t_copy);
}

