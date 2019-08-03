#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <iostream>

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

    thrust::device_ptr<int> x_ptr(x);
    thrust::device_ptr<int> y_ptr(y);
    thrust::inclusive_scan(x_ptr, x_ptr + N, y_ptr);
    y = thrust::raw_pointer_cast(y_ptr);

    int *h_y = (int*) malloc(sizeof(int) * N);
    cudaMemcpy(h_y, y, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
    {
        std::cout << h_y[i] << std::endl;
    }

    cudaFree(x);
    cudaFree(y);
    free(h_x);
    free(h_y);
    return 0;
}

