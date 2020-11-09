#ifndef _LEAK_RELU
#define _LEAK_RELU

#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void _leakyReluKer(float const *in, float *out, int size);

int reluInference(cudaStream_t stream,
                  size_t msize,
                  const void *const *inputs,
                  void **outputs);

#endif