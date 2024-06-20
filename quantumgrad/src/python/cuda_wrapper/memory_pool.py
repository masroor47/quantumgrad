import ctypes

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, '../../../lib/matrix_ops.so')
cuda_lib = ctypes.CDLL(lib_path)

class GPUMemoryPool:
    def __init__(self, chunk_size, initial_size):
        self.chunk_size = chunk_size
        self.initial_size = initial_size
        self.pool = []
        self.allocate_pool(initial_size)

    def allocate_pool(self, size):
        for _ in range(size):
            ptr = cuda_lib.allocate_gpu_memory(self.chunk_size)
            self.pool.append(ptr)

    def allocate(self, size):
        if size > self.chunk_size:
            return cuda_lib.allocate_gpu_memory(size)
        
        if len(self.pool) == 0:
            self.allocate_pool(self.initial_size)
        return self.pool.pop()

    def free(self, ptr, size):
        if size <= self.chunk_size:
            self.pool.append(ptr)
        else:
            cuda_lib.free_gpu_memory(ptr)

    def clear(self):
        while self.pool:
            ptr = self.pool.pop()
            cuda_lib.free_gpu_memory(ptr)