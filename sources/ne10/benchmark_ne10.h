#ifndef BENCHMARK_NE10_H
#define BENCHMARK_NE10_H

#include "iostream"
#include "../timer.h"
#include "libNE10_1.2.1/inc/NE10.h"

constexpr uint32_t TEST_LENGTH_SAMPLES = 32768;
constexpr uint32_t MIN_LENGTH_SAMPLES_CPX = 4;
constexpr uint32_t ARRAY_GUARD_LEN = 4;

class BenchmarkNe10 {
public:
    void run();

private:
    void fft_c2c_1d_float32_performance();
    void setup();
    void teardown();
    void createTestData();
    ne10_int32_t test_c2c_alloc (ne10_int32_t fftSize);
    uint32_t fft_c2c_1d_float32_c();
    uint32_t fft_c2c_1d_float32_neon();
    uint32_t ifft_c2c_1d_float32_c();
    uint32_t ifft_c2c_1d_float32_neon();

private:
    Timer timer;

    ne10_float32_t testInput_f32[TEST_LENGTH_SAMPLES * 2];
    ne10_fft_cfg_float32_t cfg_c;
    ne10_fft_cfg_float32_t cfg_neon;
    ne10_float32_t * guarded_in_c;
    ne10_float32_t * guarded_in_neon;
    ne10_float32_t * guarded_out_c;
    ne10_float32_t * guarded_out_neon;
    ne10_float32_t * in_c;
    ne10_float32_t * in_neon;
    ne10_float32_t * out_c;
    ne10_float32_t * out_neon;

};

#endif