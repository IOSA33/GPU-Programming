#include <omp.h>
#include <stdio.h>
#include <iostream>

int main()
{
    // --------------------------------------------- parallel single task
    int x = 1;
    #pragma omp parallel
    #pragma omp single
    {
        // out means, that change to x will occur
        #pragma omp task shared(x) depend(out: x)
            x = 2;
        // in means, that we need to wait the change to the x
        #pragma omp task shared(x) depend(in: x)
            printf("x = %d\n", x);
    }

    // --------------------------------------------- private(a)
    int a = 3;
    int b = 5;

    #pragma omp parallel private(a) shared(b)
    {
        std::cout << a << std::endl;  // undefined behavior
        std::cout << b << std::endl;  // prints 5

        #pragma omp single
        {
            b = 7; // single thread will change the value
        }

        std::cout << b << std::endl; // 7
    }

    // --------------------------------------------- private(a)
    int a1 = 3;
    int b1 = 5;

    #pragma omp parallel firstprivate(a1) shared(b1)
    {
        std::cout << a1 << std::endl;  // prints 3
        std::cout << b1 << std::endl;  // prints 5

        #pragma omp single
        {
            b = 7;  // A single thread changes the value of the variable
            a = 15; // A single thread modifies its own variable
        }

        std::cout << a << std::endl; // One thread prints 15, the rest print 3
        std::cout << b << std::endl; // 7
    }



    return 0;
}