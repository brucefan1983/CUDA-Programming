#include "error.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

int N; // number of atoms
const int NUM_REPEATS = 20; // number of timings
const int MN = 10; // maximum number of neighbors for each atom
const real cutoff = 1.9; // in units of Angstrom
const real cutoff_square = cutoff * cutoff;

void read_xy(std::vector<real>& x, std::vector<real>& y);
void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y);
void print_neighbor(const int *NN, const int *NL);

int main(void)
{
    std::vector<real> x, y;
    read_xy(x, y);
    N = x.size();
    int *NN = (int*) malloc(N * sizeof(int));
    int *NL = (int*) malloc(N * MN * sizeof(int));
    
    timing(NN, NL, x, y);
    print_neighbor(NN, NL);

    free(NN);
    free(NL);
    return 0;
}

void read_xy(std::vector<real>& v_x, std::vector<real>& v_y)
{
    std::ifstream infile("xy.txt");
    std::string line, word;
    if(!infile)
    {
        std::cout << "Cannot open xy.txt" << std::endl;
        exit(1);
    }
    while (std::getline(infile, line))
    {
        std::istringstream words(line);
        if(line.length() == 0)
        {
            continue;
        }
        for (int i = 0; i < 2; i++)
        {
            if(words >> word)
            {
                if(i == 0)
                {
                    v_x.push_back(std::stod(word));
                }
                if(i==1)
                {
                    v_y.push_back(std::stod(word));
                }
            }
            else
            {
                std::cout << "Error for reading xy.txt" << std::endl;
                exit(1);
            }
        }
    }
    infile.close();
}

void find_neighbor(int *NN, int *NL, const real* x, const real* y)
{
    for (int n = 0; n < N; n++)
    {
        NN[n] = 0;
    }

    for (int n1 = 0; n1 < N; ++n1)
    {
        real x1 = x[n1];
        real y1 = y[n1];
        for (int n2 = n1 + 1; n2 < N; ++n2)
        {
            real x12 = x[n2] - x1;
            real y12 = y[n2] - y1;
            real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square)
            {
                NL[n1 * MN + NN[n1]++] = n2;
                NL[n2 * MN + NN[n2]++] = n1;
            }
        }
    }
}

void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y)
{
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        while(cudaEventQuery(start)!=cudaSuccess){}
        find_neighbor(NN, NL, x.data(), y.data());

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
}

void print_neighbor(const int *NN, const int *NL)
{
    std::ofstream outfile("neighbor.txt");
    if (!outfile)
    {
        std::cout << "Cannot open neighbor.txt" << std::endl;
    }
    for (int n = 0; n < N; ++n)
    {
        if (NN[n] > MN)
        {
            std::cout << "Error: MN is too small." << std::endl;
            exit(1);
        }
        outfile << NN[n];
        for (int k = 0; k < MN; ++k)
        {
            if(k < NN[n])
            {
                outfile << " " << NL[n * MN + k];
            }
            else
            {
                outfile << " NaN";
            }
        }
        outfile << std::endl;
    }
    outfile.close();
}

