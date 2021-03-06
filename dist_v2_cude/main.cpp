#include "Kernel.h"
#include <stdlib.h>
#include <iostream>
#define N 64

float scale(int i, int n)
{
	return ((float)i) / (n - 1);
}

int main()
{
	const float ref = 0.5f;

	float* in = (float*)calloc(N, sizeof(float));
	float* out = (float*)calloc(N, sizeof(float));

	// compute scaled input values
	for (int i = 0; i < N; ++i)
	{
		in[i] = scale(i, N);
	}

	// compute values for the entire array
	distanceArray(out, in, ref, N);

	// release resources
	free(in);
	free(out);
	return 0;
}