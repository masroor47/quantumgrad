#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' at line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

extern "C" void* allocate_gpu_memory(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

extern "C" void free_gpu_memory(void* ptr) {
    // printf("Freeing GPU memory %p\n", ptr);
    CUDA_CHECK(cudaFree(ptr));
}

extern "C" void copy_cpu_to_gpu(const void* host_data, void* device_data, size_t size) {
    // printf("Device data: %p\n", device_data);
    // printf("Host data: %p\n", host_data);
    // print size of data to be copied in bytes
    // printf("Size: %ld\n", size);
    // print first 10 elements
    // float* data = (float*)host_data;
    // printf("CUDA _______________\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", data[i]);
    // }
    // printf("\n");

    CUDA_CHECK(cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice));
}

extern "C" void copy_gpu_to_cpu(const void* device_data, void* host_data, size_t size) {
    // print device data which is a pointer
    // printf("Device data: %p\n", device_data);
    // printf("Host data: %p\n", host_data);
    // print size of data to be copied in bytes
    // printf("Size: %ld\n", size);
    CUDA_CHECK(cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost));
    // print first 10 elements
    // float* data = (float*)host_data;
    // printf("CUDA _______________\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", data[i]);
    // }
    // printf("\n");
}