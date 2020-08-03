#include "error.cuh"
#include <stdio.h>

const unsigned WIDTH = 8;
const unsigned BLOCK_SIZE = 16;
const unsigned FULL_MASK = 0xffffffff;

void __global__ test_warp_primitives(void);

int main(int argc, char **argv)
{
    test_warp_primitives<<<1, BLOCK_SIZE>>>();
    CHECK(cudaDeviceSynchronize());
    return 0;
}

void __global__ test_warp_primitives(void)
{
    int tid = threadIdx.x;
    int lane_id = tid % WIDTH;

    if (tid == 0) printf("threadIdx.x: ");
    printf("%2d ", tid);
    if (tid == 0) printf("\n");

    if (tid == 0) printf("lane_id:     ");
    printf("%2d ", lane_id);
    if (tid == 0) printf("\n");

    unsigned mask1 = __ballot_sync(FULL_MASK, tid > 0);
    unsigned mask2 = __ballot_sync(FULL_MASK, tid == 0);
    if (tid == 0) printf("FULL_MASK = %x\n", FULL_MASK);
    if (tid == 1) printf("mask1     = %x\n", mask1);
    if (tid == 0) printf("mask2     = %x\n", mask2);

    int result = __all_sync(FULL_MASK, tid);
    if (tid == 0) printf("all_sync (FULL_MASK): %d\n", result);

    result = __all_sync(mask1, tid);
    if (tid == 1) printf("all_sync     (mask1): %d\n", result);

    result = __any_sync(FULL_MASK, tid);
    if (tid == 0) printf("any_sync (FULL_MASK): %d\n", result);

    result = __any_sync(mask2, tid);
    if (tid == 0) printf("any_sync     (mask2): %d\n", result);

    int value = __shfl_sync(FULL_MASK, tid, 2, WIDTH);
    if (tid == 0) printf("shfl:      ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    value = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) printf("shfl_up:   ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    value = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) printf("shfl_down: ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    value = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) printf("shfl_xor:  ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");
}

