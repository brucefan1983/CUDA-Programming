#include "error.cuh"
#include <iostream>
#include <vector>


void __global__ find_max(const float* d_x, float* d_y, int* d_i, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ float s_y[1024];
    __shared__ int s_i[1024];
    s_y[tid] = -1.0e30f; // a small number
    s_i[tid] = N * bid + tid;

    const int stride = 1024;
    const int number_of_rounds = (N - 1) / stride + 1;
    for (int round = 0; round < number_of_rounds; ++round) {
        const int n = round * stride + tid;
        if (n < N) {
            const int m = n + N * bid;
            float y = d_x[m];
            if (y > s_y[tid]) {
                s_y[tid] = y;
                s_i[tid] = m;
            }
        }
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            if (s_y[tid] < s_y[tid + offset]) {
                s_y[tid] = s_y[tid + offset];
                s_i[tid] = s_i[tid + offset];
            }
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_y[tid] < s_y[tid + offset]) {
                s_y[tid] = s_y[tid + offset];
                s_i[tid] = s_i[tid + offset];
            }
        }
        __syncwarp();
    }

    if (tid == 0) {
        d_y[bid] = s_y[0];
        d_i[bid] = s_i[0];
    }
}


int main(void)
{
    // problem size
    const int M = 50;
    const int N = 2000000;
    const int MN = M * N;

    // prepare host data
    std::vector<float> h_x(MN);
    for (int n = 0; n < MN; ++n) {
        h_x[n] = n;
    }

    // prepare device data
    float* d_x;
    CHECK(cudaMalloc(&d_x, MN * sizeof(float)));
    CHECK(cudaMemcpy(d_x, h_x.data(), MN * sizeof(float), cudaMemcpyHostToDevice));
    float* d_y;
    CHECK(cudaMalloc(&d_y, M * sizeof(float)));
    int* d_i;
    CHECK(cudaMalloc(&d_i, M * sizeof(int)));

    // warm up
    find_max<<<M, 1024>>>(d_x, d_y, d_i, N);

    // time
    CHECK(cudaDeviceSynchronize());
    clock_t t1 = clock();
    find_max<<<M, 1024>>>(d_x, d_y, d_i, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_t t2 = clock();
    float time_used = (t2 - t1) / float(CLOCKS_PER_SEC) * 1000.0f; // ms
    std::cout << "time used = " << time_used << " ms." << std::endl;

    // copy results to host
    std::vector<float> h_y(M);
    std::vector<int> h_i(M);
    CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_i.data(), d_i, M * sizeof(int), cudaMemcpyDeviceToHost));

    // report results
    std::cout << "max value in each row:" << std::endl;
    for (auto y : h_y) std::cout << y << std::endl;
    std::cout << std::endl;
    std::cout << "max index in each row:" << std::endl;
    for (auto i : h_i) std::cout << i << std::endl;
    std::cout << std::endl;

    // clean up
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_i));
    return 0;
}


