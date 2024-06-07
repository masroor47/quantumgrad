#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void my_cuda_kernel(int *input, int *output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        output[tid] = input[tid] * 2;
    }
}

extern "C" void my_cuda_function(int *input, int *output, int size) {
    printf("in: %i\n", input[0]);
    printf("out: %i\n", output[0]);

    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, size * sizeof(int));

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    my_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}