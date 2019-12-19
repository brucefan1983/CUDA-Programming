# Summary of my testing results

## Vector addition (chapter 5)

* Array length = 1.0e8.
* CPU function takes 77 ms and 160 ms using single and double precisions, respectively. 
* Computation times using different GPUs are listed in the table below:

|  V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | P100 (S) | P100 (D) | K40 (S) | K40 (D) |
|:---------|:---------|:---------|:---------|:---------|:---------|:---------|:---------|
| 1.5 ms | 3.0 ms |  2.1 ms |  4.3 ms | 2.2 ms |  4.3 ms | 6.5 ms | 13 ms |

* If we include cudaMemcpy, GeForce RTX 2080ti takes 130 ms and 250 ms using single and double precisions, respectively. Slower than the CPU!

## A function with high arithmetic intensity (chapter 5)
* CPU function (with an array length of 10^4) takes 320 ms and 450 ms using single and double precisions, respectively. 
* GeForce RTX 2080ti (with an array length of 10^6) takes 15 ms and 450 ms using single and double precisions, respectively.
* Tesla V100 (with an array length of 10^6) takes 11 ms and 28 ms using single and double precisions, respectively.

## Matrix copy and transpose (chapters 7 and 8)

| computation     | V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | K40 (S) |
|:---------------------------------|:-------|:-------|:-------|:-------|:-------|
| matrix copy                      | 1.1 ms | 2.0 ms | 1.6 ms | 2.9 ms |  | 
| transpose with coalesced read    | 4.5 ms | 6.2 ms | 5.3 ms | 5.4 ms | 12 ms | 
| transpose with coalesced write   | 1.6 ms | 2.2 ms | 2.8 ms | 3.7 ms | 23 ms | 
| transpose with ldg read          | 1.6 ms | 2.2 ms | 2.8 ms | 3.7 ms | 8 ms |
| transpose with bank conflict     | 1.8 ms | 2.6 ms | 3.5 ms | 4.3 ms |  | 
| transpose without bank conflict  | 1.4 ms | 2.5 ms | 2.3 ms | 4.2 ms |  |


## Reduction (chapters 8-10)

* Array length = 1.0e8.
* CPU function takes 96 ms for both single and double precisions.
* GPU timing results are listed in the following table:
| computation     | 2080ti (S) | 2080ti (D) |
|:------------|:---------|:---------|
| reduction with global memory only    | 3.2 ms | 3.8 ms |
| reduction with static shared memory  | 2.9 ms | 4.8 ms |
| reduction with dynamic shared memory | 2.9 ms | 4.8 ms |
| reduction with less blocks           | 1.0 ms | 1.7 ms |
| reduction with two kernels           | 1.0 ms | 1.6 ms | 
| reduction with atomicAdd             | 1.0 ms | 1.6 ms |
| reduction with syncwarp              | 0.9 ms | 1.6 ms | 
| reduction with shfl                  | 0.9 ms | 1.6 ms | 
| reduction with CP                    | 0.9 ms | 1.6 ms | 



## Neighbor list construction (chapter 9)

* Number of atoms = 22464.
* CPU function takes 230 ms for both single and double precisions.
* GPU timing results are list in the following table:

| computation     | V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | 
|:----------------|:---------|:---------|:-----------|:-----------|
| neighbor without atomicAdd | 2.0 ms | 2.7  ms | 1.9 ms | 17 ms |
| neighbor with atomicAdd    | 1.8 ms | 2.6  ms | 1.9 ms | 11 ms |



