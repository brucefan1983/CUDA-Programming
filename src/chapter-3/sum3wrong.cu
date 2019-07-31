#include <math.h> // fabs()
#include <stdio.h>
#define EPSILON 1.0e-14 // a small number
void __global__ sum(double *x, double *y, double *z, int N);
void check(double *z, int N);

int main(void)
{
    int N = 1024 * 100000;
    int M = sizeof(double) * N;
    // allocate host memory
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    // initialize host data
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
        y[n] = 2.0;
        z[n] = 0.0;
    }
    // allocate device memory
    double *g_x, *g_y, *g_z;
    cudaMalloc((void **)&g_x, M);
    cudaMalloc((void **)&g_y, M);
    cudaMalloc((void **)&g_z, M);
    // copy data from host to device
    cudaMemcpy(g_x, x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(g_y, y, M, cudaMemcpyHostToDevice);
    // call the kernel function
    int block_size = 128;
    int grid_size = N / block_size;
    sum<<<grid_size, block_size>>>(g_x, g_y, g_z, N);
    // copy data from device to host
    cudaMemcpy(z, g_z, M, cudaMemcpyDeviceToHost);
    // check the results
    check(z, N);
    // free host memory
    free(x);
    free(y);
    free(z);
    // gives segmentation fault
    free(g_x);
    free(g_y);
    free(g_z);
    return 0;
}

void __global__ sum(double *x, double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(double *z, int N)
{
    int has_error = 0;
    for (int n = 0; n < N; ++n)
    {
        has_error += (fabs(z[n] - 3.0) > EPSILON);
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

