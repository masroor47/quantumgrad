#include <stdio.h>
#include <cuda_runtime.h>

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

void cudaCheckError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" void* allocate_gpu_memory(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

extern "C" void free_gpu_memory(void* ptr) {
    cudaFree(ptr);
}

extern "C" void copy_to_gpu(const void* host_data, void* device_data, size_t size) {
    cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
}

extern "C" void copy_gpu_to_cpu(const void* device_data, void* host_data, size_t size) {
    cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost);
}

extern "C" void add(float *a, float *b, float *c, int rows, int cols) {
    float *d_a, *d_b, *d_c;
    size_t size = rows * cols * sizeof(float);

    cudaError_t err;

    printf("Allocating %.2f GB of GPU memory\n", (double)(size * 3) / (1024 * 1024 * 1024));
    
    err = cudaMalloc((void**)&d_a, size);
    cudaCheckError(err, "CUDA malloc failed for d_a");
    
    err = cudaMalloc((void**)&d_b, size);
    cudaCheckError(err, "CUDA malloc failed for d_b");

    err = cudaMalloc((void**)&d_c, size);
    cudaCheckError(err, "CUDA malloc failed for d_c");

    err = cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaCheckError(err, "CUDA memcpy failed for d_a");

    err = cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaCheckError(err, "CUDA memcpy failed for d_b");

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
    if (lcols != rrows) {
        printf("Matrix dimensions mismatch for multiplication\n");
        return;
    }

    float *d_left, *d_right, *d_result;
    size_t left_size = lrows * lcols * sizeof(float);
    size_t right_size = rrows * rcols * sizeof(float);
    size_t result_size = lrows * rcols * sizeof(float);

    printf("Allocating %.2f GB of GPU memory\n", (double)(left_size + right_size + result_size) / (1024 * 1024 * 1024));

    cudaError_t err;
    err = cudaMalloc((void**)&d_left, left_size);
    cudaCheckError(err, "CUDA malloc failed for d_left");

    err = cudaMalloc((void**)&d_right, right_size);
    cudaCheckError(err, "CUDA malloc failed for d_right");

    err = cudaMalloc((void**)&d_result, result_size);
    cudaCheckError(err, "CUDA malloc failed for d_result");

    err = cudaMemcpy(d_left, left, left_size, cudaMemcpyHostToDevice);
    cudaCheckError(err, "CUDA memcpy failed for d_left");

    err = cudaMemcpy(d_right, right, right_size, cudaMemcpyHostToDevice);
    cudaCheckError(err, "CUDA memcpy failed for d_right");

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