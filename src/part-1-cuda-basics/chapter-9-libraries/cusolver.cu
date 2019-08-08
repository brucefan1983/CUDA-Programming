#include "error.cuh" 
#include <stdio.h>
#include <stdlib.h>
#include <cusolverDn.h>

int main(void)
{
    int N = 2;
    int N2 = N * N;

    cuDoubleComplex *A_cpu = (cuDoubleComplex *) 
        malloc(sizeof(cuDoubleComplex) * N2);
    for (int n = 0; n < N2; ++n) 
    {
        A_cpu[0].x = 0;
        A_cpu[1].x = 0;
        A_cpu[2].x = 0;
        A_cpu[3].x = 0;
        A_cpu[0].y = 0; 
        A_cpu[1].y = 1;
        A_cpu[2].y = -1;
        A_cpu[3].y = 0;
    }
    cuDoubleComplex *A;
    CHECK(cudaMalloc((void**)&A, sizeof(cuDoubleComplex) * N2));
    CHECK(cudaMemcpy(A, A_cpu, sizeof(cuDoubleComplex) * N2, 
        cudaMemcpyHostToDevice));

    double *W_cpu = (double*) malloc(sizeof(double) * N);
    double *W; 
    CHECK(cudaMalloc((void**)&W, sizeof(double) * N));

    cusolverDnHandle_t handle = NULL;
    cusolverDnCreate(&handle);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    int lwork = 0;
    cusolverDnZheevd_bufferSize(handle, jobz, uplo, 
        N, A, N, W, &lwork);
    cuDoubleComplex* work;
    CHECK(cudaMalloc((void**)&work, 
        sizeof(cuDoubleComplex) * lwork));

    int* info;
    CHECK(cudaMalloc((void**)&info, sizeof(int)));
    cusolverDnZheevd(handle, jobz, uplo, N, A, N, W, 
        work, lwork, info);
    cudaMemcpy(W_cpu, W, sizeof(double) * N, 
        cudaMemcpyDeviceToHost);

    printf("Eigenvalues are:\n");
    for (int n = 0; n < N; ++n)
    {
        printf("%g\n", W_cpu[n]);
    }

    cusolverDnDestroy(handle);

    free(A_cpu);
    free(W_cpu);
    CHECK(cudaFree(A));
    CHECK(cudaFree(W));
    CHECK(cudaFree(work));
    CHECK(cudaFree(info));

    return 0;
}
