#ifndef ELPA_H
#define ELPA_H

#ifdef __ELPA

#include <complex>
#include <elpa/elpa_version.h>
#include <limits.h>

#if ELPA_API_VERSION < 20221101
extern "C"
{
#endif

struct elpa_struct;
typedef struct elpa_struct *elpa_t;

struct elpa_autotune_struct;
typedef struct elpa_autotune_struct *elpa_autotune_t;

#include <elpa/elpa_constants.h>
#include <elpa/elpa_generated_c_api.h>
// ELPA only provides a C interface header, causing inconsistence of complex
// between C99 (e.g. double complex) and C++11 (std::complex).
// Thus, we have to define a wrapper of complex over the c api
// for compatiability.
#if ELPA_API_VERSION < 20221101
#define complex _Complex
#endif
#include <elpa/elpa_generated.h>
#if ELPA_API_VERSION < 20221101
#undef complex
#endif
#include <elpa/elpa_generic.h>

#define ELPA_2STAGE_REAL_GPU    ELPA_2STAGE_REAL_NVIDIA_GPU
#define ELPA_2STAGE_COMPLEX_GPU ELPA_2STAGE_COMPLEX_NVIDIA_GPU

const char *elpa_strerr(int elpa_error);

#if ELPA_API_VERSION < 20221101
}//extern "C"
#endif

#endif //__ELPA
#endif //ELPA_H
