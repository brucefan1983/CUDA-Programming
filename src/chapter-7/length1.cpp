#include <stdlib.h> // malloc() and free()
#include <stdio.h> // printf()
#include <math.h> // sqrt()
#include <time.h> // clock_t, clock(), and CLOCKS_PER_SEC
double get_length(double *x, int N);

int main(void)
{
    int N = 1000;
    int M = sizeof(double) * N;
    double *x = (double *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.0;
    }
    clock_t time_begin = clock();
    double length = 0.0;
    for (int n = 0; n < 100000; ++n)
    {
        length = get_length(x, N);
    }
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
    printf("Time used for 100000 host function callings = %f s.\n", time_used);
    printf("length = %g.\n", length);
    free(x);
    return 0;
}

double get_length(double *x, int N)
{
    double length = 0.0;
    for (int n = 0; n < N; ++n)
    {
        length += x[n] * x[n];
    }
    length = sqrt(length);
    return length;
}

