#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <stdio.h>

int main(void)
{
    int N = 10;
    int *x, *y;
    cudaMalloc((void **)&x, sizeof(int) * N);
    cudaMalloc((void **)&y, sizeof(int) * N);
    int *h_x = (int*) malloc(sizeof(int) * N);
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = i + 1;
    }
    cudaMemcpy(x, h_x, sizeof(int) * N, cudaMemcpyHostToDevice);

    thrust::inclusive_scan(thrust::device, x, x + N, y);

    int *h_y = (int*) malloc(sizeof(int) * N);
    cudaMemcpy(h_y, y, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_y[i]);
    }
    printf("\n");

    cudaFree(x);
    cudaFree(y);
    free(h_x);
    free(h_y);
    return 0;
}

