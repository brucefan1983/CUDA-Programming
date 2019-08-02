#include <math.h> // fabs()
#include <stdio.h>
#include "error.cuh"
#define EPSILON 1.0e-14 // a small number
void sum(double *x, double *y, double *z, int N);
void __global__ sum(double *x, double *y, double *z, int N);
void check(double *z, int N, char* where);
void run(bool with_cpu);

int main(void)
{
    clock_t time_begin = clock();
    run(false);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin)
        / double(CLOCKS_PER_SEC);
    printf("Time used without CPU function is = %f s.\n",
        time_used);

    time_begin = clock();
    run(true);
    time_finish = clock();
    time_used = (time_finish - time_begin)
        / double(CLOCKS_PER_SEC);
    printf("Time used with CPU function is = %f s.\n",
        time_used);
    return 0;
}

void sum(double *x, double *y, double *z, int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void __global__ sum(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

void check(double *z, int N, char* where)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(z[n] - 3.0) > EPSILON);
    }
    printf("%s from %s\n", has_error ? 
        "Has errors" : "No errors", where);
}

void run(bool with_cpu)
{
    int N = 100000000;
    int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
        y[n] = 2.0;
        z[n] = 0.0;
    }
    double *g_x, *g_y, *g_z;
    CHECK(cudaMalloc((void **)&g_x, M))
    CHECK(cudaMalloc((void **)&g_y, M))
    CHECK(cudaMalloc((void **)&g_z, M))
    CHECK(cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(g_y, y, M, cudaMemcpyHostToDevice))

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;
    sum<<<grid_size, block_size>>>(g_x, g_y, g_z, N);

    if (with_cpu)
    {
        sum(x, y, z, N);
        check(z, N, "host");
    }

    CHECK(cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost))
    check(z, N, "device");
    free(x);
    free(y);
    free(z);
    CHECK(cudaFree(g_x))
    CHECK(cudaFree(g_y))
    CHECK(cudaFree(g_z))
}

