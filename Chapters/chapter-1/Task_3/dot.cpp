#include <iostream>
#include <random>
#include <vector>

double dotFunc(std::vector<double> x, std::vector<double> y, int n) {
    double dot{0};
    
    for (int i=0; i < n; i++) {
        dot = dot + x[i] * y[i];
        std::cout << "Current dot is " << dot << "\n";
    }

    return dot;
}

void addRandom(std::vector<double>& x, std::vector<double>& y, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(1,6);
    
    for (int i = 0; i < n; i++) {
        x.push_back(dist(gen));
        y.push_back(dist(gen));
    }
}


int main() {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1,6);

    int n{dist(gen)};
    std::vector<double> x;
    std::vector<double> y;

    addRandom(x, y, n);

    std::cout << "X vector: ";
    for (size_t i = 0; i < x.size(); i++) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Y vector: ";
    for (size_t i = 0; i < y.size(); i++) {
        std::cout << y[i] << " ";
    }
    std::cout << "\n";

    std::cout << dotFunc(x, y, n) << std::endl;

    return EXIT_SUCCESS;
}