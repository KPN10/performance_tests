#include "benchmark_ne10.h"

void BenchmarkNe10::run() {
    std::cout << "BenchmarkNe10 start" << std::endl;

    uint32_t time = 0;
    time = timer.milliseconds();

    fft_c2c_1d_float32_performance();

    time = timer.milliseconds() - time;
    std::cout << "time: " << time << " ms" << std::endl;
    std::cout << "BenchmarkNe10 finished" << std::endl;
}

void BenchmarkNe10::fft_c2c_1d_float32_performance() {
    createTestData();
    setup();

    uint32_t i = 0;
    uint32_t fftSize = 0;
    ne10_int32_t flag_result = NE10_OK;
    uint32_t test_loop = 0;
    uint32_t test_count = 1000;
    uint32_t time_c = 0;
    uint32_t time_neon = 0;
    float time_c_avg = 0;
    float time_neon_avg = 0;
    float time_speedup = 0;
    float time_savings = 0;

    for (fftSize = MIN_LENGTH_SAMPLES_CPX; fftSize <= TEST_LENGTH_SAMPLES; fftSize *= 2)
    {
        time_c = 0;
        time_neon = 0;
        time_c_avg = 0;
        time_neon_avg = 0;
        time_speedup = 0;
        time_savings = 0;

        flag_result = test_c2c_alloc(fftSize);
        if (flag_result == NE10_ERR) { return; }

        test_loop = test_count;


        for (i = 0; i < test_loop; i++) {
            createTestData();
            memcpy(in_c, testInput_f32, 2 * fftSize * sizeof (ne10_float32_t));
            time_c += fft_c2c_1d_float32_c();
        }

        for (i = 0; i < test_loop; i++) {
            createTestData();
            memcpy(in_neon, testInput_f32, 2 * fftSize * sizeof (ne10_float32_t));
            time_neon += fft_c2c_1d_float32_neon();
        }

        time_speedup = (float) time_c / time_neon;
        time_savings = ( ( (float) (time_c - time_neon)) / time_c) * 100;
        time_c_avg = (float) time_c / test_loop;
        time_neon_avg = (float) time_neon / test_loop;

        std::cout << "FFT size: " << fftSize << "; "
        << "time_c: " << time_c << "; " << "time_c_avg: " << time_c_avg << "; "
        << "time_neon: " << time_neon << "; " << "time_neon_avg: " << time_neon_avg << "; "
        << "time_saving: " << time_savings << "; " << "time_speedup: " << time_speedup << ";" << std::endl;
    }

    teardown();
}

ne10_int32_t BenchmarkNe10::test_c2c_alloc (ne10_int32_t xFftSize) {
    // NE10_FREE (cfg_c);
    // NE10_FREE (cfg_neon);

    cfg_c = ne10_fft_alloc_c2c_float32_c(xFftSize);
    if (cfg_c == NULL) {
        return NE10_ERR;
    }

    cfg_neon = ne10_fft_alloc_c2c_float32_neon(xFftSize);
    if (cfg_neon == NULL) {
        return NE10_ERR;
    }

    return NE10_OK;
}

void BenchmarkNe10::setup() {
    /* init input memory */
    guarded_in_c = (ne10_float32_t*) NE10_MALLOC ( (TEST_LENGTH_SAMPLES * 2 + ARRAY_GUARD_LEN * 2) * sizeof (ne10_float32_t));
    guarded_in_neon = (ne10_float32_t*) NE10_MALLOC ( (TEST_LENGTH_SAMPLES * 2 + ARRAY_GUARD_LEN * 2) * sizeof (ne10_float32_t));
    in_c = guarded_in_c + ARRAY_GUARD_LEN;
    in_neon = guarded_in_neon + ARRAY_GUARD_LEN;

    /* init dst memory */
    guarded_out_c = (ne10_float32_t*) NE10_MALLOC ( (TEST_LENGTH_SAMPLES * 2 + ARRAY_GUARD_LEN * 2) * sizeof (ne10_float32_t));
    guarded_out_neon = (ne10_float32_t*) NE10_MALLOC ( (TEST_LENGTH_SAMPLES * 2 + ARRAY_GUARD_LEN * 2) * sizeof (ne10_float32_t));
    out_c = guarded_out_c + ARRAY_GUARD_LEN;
    out_neon = guarded_out_neon + ARRAY_GUARD_LEN;
}

void BenchmarkNe10::teardown() {
    NE10_FREE (cfg_c);
    NE10_FREE (cfg_neon);
    NE10_FREE (guarded_in_c);
    NE10_FREE (guarded_in_neon);
    NE10_FREE (guarded_out_c);
    NE10_FREE (guarded_out_neon);
}

void BenchmarkNe10::createTestData() {
    for (uint32_t i = 0; i < TEST_LENGTH_SAMPLES * 2; i++)
    {
        testInput_f32[i] = (ne10_float32_t) (drand48() * 32768.0f - 16384.0f);
    }
}

uint32_t BenchmarkNe10::fft_c2c_1d_float32_c() {
    uint32_t time = timer.nanoseconds();
    ne10_fft_c2c_1d_float32_c( (ne10_fft_cpx_float32_t*) out_c, (ne10_fft_cpx_float32_t*) in_c, cfg_c, 0 );
    return timer.nanoseconds() - time;
}

uint32_t BenchmarkNe10::fft_c2c_1d_float32_neon() {
    uint32_t time = timer.nanoseconds();
    ne10_fft_c2c_1d_float32_neon( (ne10_fft_cpx_float32_t*) out_neon, (ne10_fft_cpx_float32_t*) in_neon, cfg_neon, 0 );
    return timer.nanoseconds() - time;
}

uint32_t BenchmarkNe10::ifft_c2c_1d_float32_c() {
    uint32_t time = timer.nanoseconds();
    ne10_fft_c2c_1d_float32_c( (ne10_fft_cpx_float32_t*) out_c, (ne10_fft_cpx_float32_t*) in_c, cfg_c, 1 );
    return timer.nanoseconds() - time;
}

uint32_t BenchmarkNe10::ifft_c2c_1d_float32_neon() {
    uint32_t time = timer.nanoseconds();
    ne10_fft_c2c_1d_float32_neon( (ne10_fft_cpx_float32_t*) out_neon, (ne10_fft_cpx_float32_t*) in_neon, cfg_neon, 1 );
    return timer.nanoseconds() - time;
}