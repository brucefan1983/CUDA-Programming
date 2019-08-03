#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <iostream>

int main(void)
{
    int N = 10;
    thrust::device_vector<int> x(N, 0);
    thrust::device_vector<int> y(N, 0);
    for (int i = 0; i < x.size(); ++i)
    {
        x[i] = i + 1;
    }
    thrust::inclusive_scan(x.begin(), x.end(), y.begin());
    for (int i = 0; i < y.size(); ++i)
    {
        std::cout << y[i] << std::endl;
    }
    return 0;
}

