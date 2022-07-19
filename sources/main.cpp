#include <iostream>
#include "ne10/benchmark_ne10.h"

int main() {
    std::cout<<"Performance tests"<<std::endl;
    BenchmarkNe10 benchmarkNe10;
    benchmarkNe10.run();

    return 0;
}