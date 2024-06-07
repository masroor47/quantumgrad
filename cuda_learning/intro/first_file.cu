#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// macro helper for error checking
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return -1; \
    }


__global__ void addVectors(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

void printArr(int *arr, int size) {
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << "... ";
    for (int i = size - 5; i < size; i++) {
        std::cout << arr[i] << ", "; 
    }
    std::cout << std::endl;
}

int main() {
    // Anything larger than 1 causes a very slow host memory allocation
    long long int GBs = 1;
    const size_t size = GBs * 1024 * 1024 * 1024 / 4;
    int* A, * B, * C;
    double GB_rounded = 3 * static_cast<double>(size) * sizeof(int) / (1024 * 1024 * 1024);
    std::cout << "Trying to allocate " << GB_rounded << "GB in total. First on host then on device" << std::endl; 

    cudaMallocHost(&A, size * sizeof(int));
    cudaMallocHost(&B, size * sizeof(int));
    cudaMallocHost(&C, size * sizeof(int));

    std::cout << "Allocated on host!\n" << std::endl;

    for (int i = 0; i < size; i++) {
        A[i] = i;
        B[i] = i * 2;
    }
    
    std::cout << "A: ";
    printArr(A, size);
    std::cout << "B: ";
    printArr(B, size);
    std::cout << "C: ";
    printArr(C, size);

    std::cout << "Initialized values\n" << std::endl;

    int* d_A, * d_B, * d_C;

    cudaError_t err = cudaMalloc(&d_A, size * sizeof(int));
    CUDA_CHECK(err);

    err = cudaMalloc(&d_B, size * sizeof(int));
    CUDA_CHECK(err);

    err = cudaMalloc(&d_C, size * sizeof(int));
    CUDA_CHECK(err);

    std::cout << "Allocated on device\n" << std::endl;
    
    std::cout << "Moving to device..." << std::endl;
    cudaMemcpy(d_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Moved to device\n" << std::endl;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    std::cout << "Launching kernel..." << std::endl;
    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(d_A, d_B, d_C, size);


    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(kernelErr) << std::endl;

    }
    std::cout << "Computed successfully\n" << std::endl;

    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, d_C, size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Kernel execution time: " << milliseconds << " ms\n" << std::endl;

    std::cout << "C: ";
    printArr(C, size);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Freed on device\n" << std::endl; 
    
    std::cout << "Freeing on host..." << std::endl; 
    // Free host memory
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    std::cout << "Freed on host\n" << std::endl; 

    return 0;
}
