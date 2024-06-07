#include <stdio.h>

__global__ void add_matrices(float *a, float *b, float *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void matrix_add(float *a, float *b, float *c, int rows, int cols) {
    float *d_a, *d_b, *d_c;

    printf("Allocating %.2f GB of GPU memory\n", (double)(rows * cols * sizeof(float) * 3) / (1024 * 1024 * 1024));
    cudaMalloc((void**)&d_a, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_b, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_c, rows * cols * sizeof(float));

    cudaMemcpy(d_a, a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    add_matrices<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, rows, cols);

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(kernelErr));
    }

    cudaMemcpy(c, d_c, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}