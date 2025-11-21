#include <iostream>
#include <cmath>
#include "Timer.h"
#include <omp.h>
#include <stdio.h>
#include <random>
#include <array>

#define N 3000
#define A -1
#define B 1

using namespace std;

double matrixWeightedDot(double *Matrice, double *x, double *y, int n) {
    double dot = 0;

    #pragma omp parallel for reduction(+:dot) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dot += Matrice[i*n+j] * x[i] * y[j];
        }
    }

    return dot;
}

int main() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<> dist(A, B);

    // Allocate an N*M array
    double *array = new double[N*N];
    double *x = new double[N];
    double *y = new double[N];

    for (int i = 0; i < N; i++) {
        x[i] = dist(gen);
        y[i] = dist(gen);
        for (int j = 0; j < N; j++)
            array[i * N + j] = dist(gen);
    }

    std::array<double, 12> arr{};

    for (int threads = 1; threads <= 12; ++threads) {
        std::cout << std::endl;
        std::cout << "threads: " << threads << std::endl;
        omp_set_num_threads(threads);

        Timer timer;
        timer.begin();

        const double value = matrixWeightedDot(array, x, y, N);

        timer.end();
        cout << "Time: " << timer.getTime() << " s" << endl;

        // N yhteenlaskua
        // N additions
        cout << "Flops: " << 1.0E-9*N/timer.getTime() << " GFlops" << endl;

        double realValue = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                realValue += array[i*N+j] * x[i] * y[j];
            }
        }

        cout << "Sum value: " << value << std::endl;
        cout << "Real value: " << realValue << endl;
        cout << "Diff: " << fabs(value-realValue) << std::endl;
        arr[threads - 1] = timer.getTime();
    }


    for (int i = 1; i < arr.size() + 1; i++) {
        cout << "Time is: " << (arr[0]/(i*arr[i-1]))*100/100 << endl;
    }

    // Free the N*M array
    delete [] array;
    delete [] x;
    delete [] y;

    return 0;
}