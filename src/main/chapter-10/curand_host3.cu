#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
void output_results(int N, double *g_x);

int main(void)
{
    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234);
    int N = 100000;
    double *x = (double*) malloc(N * sizeof(double));
    curandGenerateNormalDouble(generator, x, N, 0.0, 1.0);
    output_results(N, x);
    free(x);
    return 0;
}

void output_results(int N, double *x)
{
    FILE *fid = fopen("x3.txt", "w");
    for(int n = 0; n < N; n++)
    {
        fprintf(fid, "%g\n", x[n]);
    }
    fclose(fid);
}

