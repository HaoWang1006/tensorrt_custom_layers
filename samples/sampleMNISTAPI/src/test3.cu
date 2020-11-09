#include <iostream>
#include <time.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add(int *a, int *b, int *c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while(index < 20)
  {
    c[index] =  a[index] + b[index];
    index += blockDim.x * gridDim.x;
    printf("%d\n",gridDim.x);
  }
}

int main()
{
  int n = 20;
  int *a = (int *)malloc(sizeof(int)*n);
  int *b = (int *)malloc(sizeof(int)*n);
  int *c = (int *)malloc(sizeof(int)*n);

  int *dev_a;
  int *dev_b;
  int *dev_c;
  int *dev_n;

  // 
  for(int i=0;i<n;i++)
  {
    a[i] = i;
    b[i] = i;
  }

  std::cout << ">>>> 1 <<<<" <<std::endl;

  cudaMalloc((void**)&dev_a, sizeof(int)*n);
  cudaMalloc((void**)&dev_b, sizeof(int)*n);
  cudaMalloc((void**)&dev_c, sizeof(int)*n);
  cudaMalloc((void**)&dev_n, sizeof(int));
  // add<<<1,1>>>(2,7,dev_c);

  std::cout << ">>>> 2 <<<<" <<std::endl;

  cudaMemcpy(dev_a,a,sizeof(int)*n,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b,b,sizeof(int)*n,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_n,&n,sizeof(int),cudaMemcpyHostToDevice);

  std::cout << ">>>> 3 <<<<" <<std::endl;

  dim3 blockSize(20);
  dim3 gridSize((n+blockSize.x-1)/blockSize.x); //1
  // add<<<grid，block>>>
  // gridSize個線程塊*blockSize個線程/線程塊
  // add<<<gridSize, blockSize>>>(dev_a,dev_b,dev_c);
  add<<<1, 1>>>(dev_a,dev_b,dev_c);

  std::cout << ">>>> 4 <<<<" <<std::endl;

  cudaMemcpy(c,dev_c,sizeof(int)*n,cudaMemcpyDeviceToHost);

  for(int i=0;i<n;i++)
  {
    printf("c[%d] = %d\n",i,c[i]);
  }
  
  cudaFree(dev_c);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_n);
  
  free(a);
  free(b);
  free(c);

  std::cout << ">>>> 5 <<<<" <<std::endl;

  return 0;
}
