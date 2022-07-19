#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "NE10.h"
#include <sys/time.h>
#include "current_time.h"

using namespace std;

/* Max FFT Length and double buffer for real and imag */
#define TEST_LENGTH_SAMPLES (32768)
#define MIN_LENGTH_SAMPLES_CPX (4)
#define MIN_LENGTH_SAMPLES_REAL (MIN_LENGTH_SAMPLES_CPX*2)

//input and output
static ne10_float32_t * guarded_in_c = NULL;
static ne10_float32_t * guarded_in_neon = NULL;
static ne10_float32_t * in_c = NULL;
static ne10_float32_t * in_neon = NULL;

static ne10_float32_t * guarded_out_c = NULL;
static ne10_float32_t * guarded_out_neon = NULL;
static ne10_float32_t * out_c = NULL;
static ne10_float32_t * out_neon = NULL;

static ne10_float32_t snr = 0.0f;

static ne10_int64_t time_c = 0;
static ne10_int64_t time_neon = 0;
static ne10_float32_t time_speedup = 0.0f;
static ne10_float32_t time_savings = 0.0f;

static ne10_fft_cfg_float32_t cfg_c;
static ne10_fft_cfg_float32_t cfg_neon;

static ne10_int32_t test_c2c_alloc (ne10_int32_t fftSize);
ne10_int32_t test_c2c_alloc (ne10_int32_t fftSize)
{
    NE10_FREE (cfg_c);
    NE10_FREE (cfg_neon);

    cfg_c = ne10_fft_alloc_c2c_float32_c (fftSize);
    if (cfg_c == NULL)
    {
        fprintf (stdout, "======ERROR, FFT alloc fails\n");
        return NE10_ERR;
    }

    cfg_neon = ne10_fft_alloc_c2c_float32_neon (fftSize);
    if (cfg_neon == NULL)
    {
        NE10_FREE (cfg_c);
        fprintf (stdout, "======ERROR, FFT alloc fails\n");
        return NE10_ERR;
    }
    return NE10_OK;
}


static ne10_float32_t testInput_f32[TEST_LENGTH_SAMPLES * 2];

long int GetTickCount()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000000 + tv.tv_usec;
}

// #define GET_TIME(time, code) { \
//     (time) = GetTickCount(); \
//     code \
//     (time) = GetTickCount() - (time);\
// }

CurrentTime now;
#define GET_TIME(time, code) { \
        (time) = now.nanoseconds(); \
        code \
        (time) = now.nanoseconds() - (time);\
    }

// #define GET_TIME(time, code) { \
//         (time) = now.microseconds(); \
//         code \
//         (time) = now.microseconds() - (time);\
//     }

char ne10_log_buffer[5000];
char *ne10_log_buffer_ptr;
void ne10_log(const char *func_name,
              const char *format_str,
              ne10_int32_t n,
              ne10_int32_t time_c,
              ne10_int32_t time_neon,
              ne10_float32_t time_savings,
              ne10_float32_t time_speedup)
{
    int byte_count = 0;
    byte_count = sprintf(ne10_log_buffer_ptr,
                         "{ \"name\" : \"%s %d\", \"time_c\" : %d, "
                         "\"time_neon\" : %d },",
                         func_name, n, time_c, time_neon);
    ne10_log_buffer_ptr += byte_count;

    /* print the result, which is needed by command line performance test. */
    fprintf (stdout,
             "%25d%20d%20d%19.2f%%%18.2f:1\n",
             n,
             time_c,
             time_neon,
             time_savings,
             time_speedup);
}

void test_fft_c2c_1d_float32_performance(uint32_t test_count);

#define ARRAY_GUARD_LEN      4
static void test_setup (void)
{
    ne10_log_buffer_ptr = ne10_log_buffer;
    ne10_int32_t i;

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

    for (i = 0; i < TEST_LENGTH_SAMPLES * 2; i++)
    {
        testInput_f32[i] = (ne10_float32_t) (drand48() * 32768.0f - 16384.0f);
    }
}

static void test_teardown (void)
{
    NE10_FREE (guarded_in_c);
    NE10_FREE (guarded_in_neon);
    NE10_FREE (guarded_out_c);
    NE10_FREE (guarded_out_neon);
}

int main()
{
    CurrentTime now;
    uint64_t time;
    uint32_t test_count;
    cout << "Start perfomance test" << endl;
    time = now.milliseconds();

    test_setup();
    test_count = 1;
    cout << "TEST_COUNT: " << test_count << endl;
    test_fft_c2c_1d_float32_performance(test_count);
    test_teardown();

    cout << "elapsed time: " << now.milliseconds() - time << " ms" << endl;
    cout<<"==========end========="<<endl;
    return 0;
}

void test_fft_c2c_1d_float32_performance(uint32_t test_count)
{
    ne10_int32_t i = 0;
    ne10_int32_t fftSize = 0;
    ne10_int32_t flag_result = NE10_OK;
    ne10_int32_t test_loop = 0;

    fprintf (stdout, "----------%30s\n", __FUNCTION__);
    fprintf (stdout, "%25s%20s%20s%20s%20s\n", "FFT Length", "C Time in ns", "NEON Time in ns", "Time Savings", "Performance Ratio");

    for (fftSize = MIN_LENGTH_SAMPLES_CPX; fftSize <= TEST_LENGTH_SAMPLES; fftSize *= 2)
    {
        fprintf (stdout, "FFT size %d\n", fftSize);

        /* FFT test */
        memcpy (in_c, testInput_f32, 2 * fftSize * sizeof (ne10_float32_t));
        memcpy (in_neon, testInput_f32, 2 * fftSize * sizeof (ne10_float32_t));
        flag_result = test_c2c_alloc (fftSize);
        if (flag_result == NE10_ERR)
        {
            return;
        }

        test_loop = test_count / fftSize;

        GET_TIME
        (
            time_c,
        {
            for (i = 0; i < test_loop; i++)
                ne10_fft_c2c_1d_float32_c ( (ne10_fft_cpx_float32_t*) out_c, (ne10_fft_cpx_float32_t*) in_c, cfg_c, 0);
        }
        );
        GET_TIME
        (
            time_neon,
        {
            for (i = 0; i < test_loop; i++)
                ne10_fft_c2c_1d_float32_neon ( (ne10_fft_cpx_float32_t*) out_neon, (ne10_fft_cpx_float32_t*) in_neon, cfg_neon, 0);
        }
        );

        time_speedup = (ne10_float32_t) time_c / time_neon;
        time_savings = ( ( (ne10_float32_t) (time_c - time_neon)) / time_c) * 100;
        ne10_log (__FUNCTION__, "Float FFT%21d%20lld%20lld%19.2f%%%18.2f:1\n", fftSize, time_c, time_neon, time_savings, time_speedup);

        /* IFFT test */
        memcpy (in_c, out_c, 2 * fftSize * sizeof (ne10_float32_t));
        memcpy (in_neon, out_c, 2 * fftSize * sizeof (ne10_float32_t));

        GET_TIME
        (
            time_c,
        {
            for (i = 0; i < test_loop; i++)
                ne10_fft_c2c_1d_float32_c ( (ne10_fft_cpx_float32_t*) out_c, (ne10_fft_cpx_float32_t*) in_c, cfg_c, 1);
        }
        );
        GET_TIME
        (
            time_neon,
        {
            for (i = 0; i < test_loop; i++)
                ne10_fft_c2c_1d_float32_neon ( (ne10_fft_cpx_float32_t*) out_neon, (ne10_fft_cpx_float32_t*) in_neon, cfg_neon, 1);
        }
        );

        time_speedup = (ne10_float32_t) time_c / time_neon;
        time_savings = ( ( (ne10_float32_t) (time_c - time_neon)) / time_c) * 100;
        ne10_log (__FUNCTION__, "Float FFT%21d%20lld%20lld%19.2f%%%18.2f:1\n", fftSize, time_c, time_neon, time_savings, time_speedup);

        NE10_FREE (cfg_c);
        NE10_FREE (cfg_neon);
    }
}
