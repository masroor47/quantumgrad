#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void* allocate_gpu_memory(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

extern "C" void free_gpu_memory(void* ptr) {
    cudaFree(ptr);
}

extern "C" void copy_cpu_to_gpu(const void* host_data, void* device_data, size_t size) {
    printf("Device data: %p\n", device_data);
    // print size of data to be copied in bytes
    printf("Size: %ld\n", size);

    cudaError_t err = cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    // print first 10 elements
    float* data = (float*)host_data;
    printf("CUDA _______________\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
}

extern "C" void copy_gpu_to_cpu(const void* device_data, void* host_data, size_t size) {
    // print device data which is a pointer
    printf("Device data: %p\n", device_data);
    cudaError_t err = cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    // print first 10 elements
    float* data = (float*)host_data;
    printf("CUDA _______________\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
}