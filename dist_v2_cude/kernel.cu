
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Kernel.h"
#include <stdio.h>
#include <math.h>
#define TPB 32

__device__ float scale(int i, int n)
{
	return ((float)i) / (n - 1);
}

__device__ float distance(float x1, float x2)
{
	return sqrt((x2 - x1) * (x2 - x1));
}

// converted from serial app
__global__ void distanceKernel(float* d_out, float ref, int len)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const float x = scale(i, len);
	d_out[i] = distance(x, ref);
	printf("i = %2d: dist from %f to %f.\n", i, ref, x, d_out[i]);
}

__global__ void distanceArray(float* out, float* in, float ref, int len)
{
	// Declare pointers to device arrays
	float* d_in = 0;
	float* d_out = 0;

	// Allowcate memory for device arrays
	cudaMalloc(&d_in, len * sizeof(float));
	cudaMalloc(&d_out, len * sizeof(float));

	// Copy input data from host to device
	cudaMemcpy(d_in, in, len * sizeof(float), cudaMemcpyHostToDevice);

	// launch kernel to compute and store distance values
	distanceKernel<<<len / TPB, TPB >>>(d_out, d_in, ref);

	// Copy results from device to host
	cudaMemcpy(out, d_out, len * sizeof(float), cudaMemcpyDeviceToHost);

	// free the memory allowcated for device arrays
	cudaFree(d_in);
	cudaFree(d_out);
}

