#include <stdio.h>

__global__ void add_matrices(float *a, float *b, float *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matmul_simple_(float *left, float *right, float *result, int lrows, int lcols, int rrows, int rcols) {
    // current row of left, current col of right
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < lrows && col < rcols) {
        float temp_sum = 0.0;
        // dot product
        for (int i = 0; i < lcols; i++) {
            temp_sum += left[row * lcols + i] * right[i * rcols + col];
        }
        result[row * rcols + col] = temp_sum;
    }
}

extern "C" void add(float *a, float *b, float *c, int rows, int cols) {
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

extern "C" void matmul_simple(float *left, float *right, float *result, int lrows, int lcols, int rrows, int rcols) {
    float *d_left, *d_right, *d_result;
    size_t left_size = lrows * lcols * sizeof(float);
    size_t right_size = rrows * rcols * sizeof(float);
    size_t result_size = lrows * rcols * sizeof(float);

    double total_gb = (left_size + right_size + result_size) / (1024 * 1024 * 1024);
    printf("Allocating %.2f GB of GPU memory\n", total_gb);

    cudaError_t err;
    err = cudaMalloc((void**)&d_left, left_size);
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_left: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc((void**)&d_right, right_size);
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_right: %s\n", cudaGetErrorString(err));
        cudaFree(d_left);
        return;
    }

    err = cudaMalloc((void**)&d_result, result_size);
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_result: %s\n", cudaGetErrorString(err));
        cudaFree(d_left);
        cudaFree(d_right);
        return;
    }

    err = cudaMemcpy(d_left, left, left_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed for d_left: %s\n", cudaGetErrorString(err));
        cudaFree(d_left);
        cudaFree(d_right);
        cudaFree(d_result);
        return;
    }

    err = cudaMemcpy(d_right, right, right_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed for d_right: %s\n", cudaGetErrorString(err));
        cudaFree(d_left);
        cudaFree(d_right);
        cudaFree(d_result);
        return;
    }

    int BLOCK_SIZE = 16;
    int GRID_SIZE_ROWS = (int)ceil((float)lrows / BLOCK_SIZE);
    int GRID_SIZE_COLS = (int)ceil((float)rcols / BLOCK_SIZE);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(GRID_SIZE_COLS, GRID_SIZE_ROWS);

    matmul_simple_<<<gridSize, blockSize>>>(d_left, d_right, d_result, lrows, lcols, rrows, rcols);

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(kernelErr));
    }

    cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_result);
}