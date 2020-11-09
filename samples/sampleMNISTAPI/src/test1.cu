#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// https://www.cnblogs.com/zl1991/p/12449662.html

// 矩阵类型，行优先，M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
  int width;
  int height;
  float *elements;
};

__global__ void add(float *x, float *y, float *z, int n)
{
  // 获取全局索引
  // threadIdx.x， x direction thread id number
  // blockIdx.x,   x direction block id number
  // blockDim.x,   x direction Dimension id number, max value

  int index = threadIdx.x + blockIdx.x * blockDim.x; // 3 + 2 * 256
  // printf("%d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
  // 步长
  int stride = blockDim.x * gridDim.x; //需要计算的数组很大

  for (int i = index; i < n; i += stride)
  {
    // printf("@@@@@@@@@@@@@@@@@@@@ ==== %d\n", stride);
    z[i] = x[i] + y[i]; // 不会被执行，即每个线程只处理一个加法。可实现为每个线程处理多个加法；
  }
}

// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix *A, int row, int col)
{
  return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix *A, int row, int col, float value)
{
  A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C)
{
  float Cvalue = 0.0;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = 0; i < A->width; ++i)
  {
    Cvalue += getElement(A, row, i) * getElement(B, i, col);
  }
  setElement(C, row, col, Cvalue);
}

int main()
{
  int N = 1 << 20;
  int nBytes = N * sizeof(float);

  // 申请_device_设备端内存, 不能在主机端执行写操作
  float *x, *y, *z;
  cudaMallocManaged((void **)&x, nBytes);
  cudaMallocManaged((void **)&y, nBytes);
  cudaMallocManaged((void **)&z, nBytes);

  // 初始化数据
  for (int i = 0; i < N; ++i)
  {
    x[i] = 10.0;
    y[i] = 20.0;
  }

  // 定义kernel的执行配置
  dim3 blockSize(256);
  dim3 gridSize(4096); // number of blocks
  std::cout << "(N + blockSize.x - 1) / blockSize.x === " << (N + blockSize.x - 1) / blockSize.x << std::endl;
  // 执行kernel
  add<<<gridSize, blockSize>>>(x, y, z, N);

  // 同步device 保证结果能正确访问
  cudaDeviceSynchronize();

  // 检查执行结果
  float maxError = 0.0;
  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(z[i] - 30.0));
  }

  std::cout << "Max error: " << maxError << std::endl;

  // 释放内存
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);

  return 0;
}

// int main()
// {
//   int width = 1 << 10;
//   int height = 1 << 10;
//   Matrix *A, *B, *C;

//   // 申请托管内存
//   std::cout << "sizeof(Matrix) == " << sizeof(Matrix) << std::endl;
//   cudaMallocManaged((void **)&A, sizeof(Matrix));
//   cudaMallocManaged((void **)&B, sizeof(Matrix));
//   cudaMallocManaged((void **)&C, sizeof(Matrix));
//   int nBytes = width * height * sizeof(float);
//   cudaMallocManaged((void **)&A->elements, nBytes);
//   cudaMallocManaged((void **)&B->elements, nBytes);
//   cudaMallocManaged((void **)&C->elements, nBytes);

//   // 初始化数据
//   A->height = height;
//   A->width = width;
//   B->height = height;
//   B->width = width;
//   C->height = height;
//   C->width = width;
//   for (int i = 0; i < width * height; ++i)
//   {
//     A->elements[i] = 1.0;
//     B->elements[i] = 2.0;
//   }

//   // 定义kernel的执行配置
//   dim3 blockSize(32, 32);
//   dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
//                 (height + blockSize.y - 1) / blockSize.y); // (32, 32)

//   std::cout << "(width + blockSize.x - 1) / blockSize.x === " << (width + blockSize.x - 1) / blockSize.x << std::endl;
//   std::cout << "(height + blockSize.y - 1) / blockSize.y === " << (height + blockSize.y - 1) / blockSize.y << std::endl;
//   // 执行kernel
//   matMulKernel<<<gridSize, blockSize>>>(A, B, C);

//   // 同步device 保证结果能正确访问
//   cudaDeviceSynchronize();
//   // 检查执行结果
//   float maxError = 0.0;
//   for (int i = 0; i < width * height; ++i)
//   {
//     maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
//   }

//   std::cout << "Max error: " << maxError << std::endl;

//   return 0;
// }