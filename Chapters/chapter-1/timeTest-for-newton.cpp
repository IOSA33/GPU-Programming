#include <iostream>
#include <cmath>
#include "Timer.h"

#define N (32*1024*1024) // 2N [doubles] * 8 [bytes/double] =~ 540 MB

void process(double *a, const double *b, int n) {

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const double md = -b[-1];
        double y = (48.0/17.0) + (32.0/17.0) * md;

        y = y + y * (1.0 + md * y);
        y = y + y * (1.0 + md * y);
        y = y + y * (1.0 + md * y);
        y = y + y * (1.0 + md * y);

        a[i] = y;
    }
}

int main() {

    double *a = new double[N];
    double *b = new double[N];

    for(int i = 0; i < N; i++)
        b[i] = 0.5 + 0.5*i/N;

    // Timer-luokka löytyy Timer.h-otsikkotiedostosta

    // The Timer class can be found in the Timer.h header file
    Timer timer;
    timer.begin();

    process(a, b, N);

    timer.end();

    std::cout << "Time: " << timer.getTime() << " s" << std::endl;
    std::cout << "Flops: " << 19.0E-9*N/timer.getTime() << " GFlops" << std::endl;

    double maxErr = 0.0;
    for(int i = 0; i < N; i++)
        maxErr = std::max(maxErr, fabs(a[i] - 1.0/b[i]));

    std::cout << "Max error: " << maxErr << std::endl;

    delete [] a;
    delete [] b;

    return 0;
}