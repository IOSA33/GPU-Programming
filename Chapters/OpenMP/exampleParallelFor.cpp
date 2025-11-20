// This is non-parallel func, kinda slow
double sum(double *x, int n) {
    double tmp = 0.0;
    for(int i = 0; i < n; i++)
        tmp += x[i];
    return tmp;
}

// The program produces the correct result
// and is almost 20 times faster than the unparallelized version!
double sumThread(double *x, int n) {
    double tmp = 0.0;

    #pragma omp parallel
    {
        double p = 0.0;

        #pragma omp for nowait
        for(int i = 0; i < n; i++)
            p += x[i];

        // Best is we don't put atomic operation inside the loop
        #pragma omp atomic
        tmp += p;
    }

    return tmp;
}

// same as above, but in just one line
double sumThreadOneLine(double *x, int n) {
    double tmp = 0.0;

    #pragma omp parallel for reduction(+:tmp)
    for(int i = 0; i < n; i++) {
        tmp += x[i];
    }

    return tmp;
}

int main() {
    return 0;
}