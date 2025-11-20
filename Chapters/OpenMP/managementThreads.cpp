#include <iostream>
#include <stdio.h>

int main() {
    // We are choosing how much parallel threads we want
    #pragma omp parallel for num_threads(6)
    for(int i = 0; i < n; ++i) {
        a[i] = i;
    }

    // --------------------------------- Synchronisation
    #pragma omp parallel
    {

        subroutine1();

        // Every thread should wait here that every thread is completed above
        #pragma omp barrier

        // subrountine2 below only starts executing after previous barrier has
        // been reached by all threads
        subroutine2();

    }

}
