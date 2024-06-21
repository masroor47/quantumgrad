#include <cuda_runtime.h>

extern "C" void* allocate_gpu_memory(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

extern "C" void free_gpu_memory(void* ptr) {
    cudaFree(ptr);
}

extern "C" void copy_cpu_to_gpu(const void* host_data, void* device_data, size_t size) {
    cudaMemcpy(device_data, host_data, size * sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" void copy_gpu_to_cpu(const void* device_data, void* host_data, size_t size) {
    cudaMemcpy(host_data, device_data, size*sizeof(float), cudaMemcpyDeviceToHost);
}