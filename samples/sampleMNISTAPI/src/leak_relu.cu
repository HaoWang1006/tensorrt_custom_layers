#include "leak_relu.h"

__global__ void _leakyReluKer(float const *in, float *out, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index >= size)
  {
    return;
  }

  if (in[index] < 0)
  {
    out[index] = in[index] * 0.1;
  }
  else
  {
    out[index] = in[index];
  }
}

int reluInference(cudaStream_t stream, size_t msize, const void *const *inputs, void **outputs)
{
  int block_size = 256;
  
  int grid_size = (msize + block_size - 1) / block_size;
  printf("block_size =============== %d\n", block_size);
  printf("grid_size =============== %d\n", grid_size);
  printf("msize =============== %d\n", msize);//1*28*28
  
  _leakyReluKer<<<grid_size, block_size>>>(
      reinterpret_cast<float const *>(inputs[0]),
      reinterpret_cast<float *>(outputs[0]), msize); 
}
