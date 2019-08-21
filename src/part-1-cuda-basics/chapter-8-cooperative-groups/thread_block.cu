#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void hello_from_gpu(void)
{
    cg::thread_block g = cg::this_thread_block();
    printf("Hello World from block-%d and thread-(%d, %d)!\n", 
        g.group_index().x, g.thread_index().x, 
        g.thread_index().y);
    if (g.thread_rank() == 1)
    {
        printf("Block size = %d\n", g.size());
    }
}
int main(void)
{
    dim3 block_size(1, 4);
    hello_from_gpu<<<2, block_size>>>();
    cudaDeviceReset();
    return 0;
}
