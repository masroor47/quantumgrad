import os
import ctypes
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_lib_path = os.path.join(current_dir, '../../lib/matrix_ops.so')
cuda_lib = ctypes.cdll.LoadLibrary(cuda_lib_path)

cuda_lib.add.argtypes = [ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int]

cuda_lib.matmul_simple.argtypes = [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int]

# Define the return types and argument types for the CUDA functions
cuda_lib.allocate_gpu_memory.restype = ctypes.c_void_p
cuda_lib.allocate_gpu_memory.argtypes = [ctypes.c_size_t]

cuda_lib.free_gpu_memory.restype = None
cuda_lib.free_gpu_memory.argtypes = [ctypes.c_void_p]

cuda_lib.copy_to_gpu.restype = None
cuda_lib.copy_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

def cpu_to_gpu(data):
    print('cuda interface: moving cpu to gpu')
    gpu_data = cuda_lib.allocate_gpu_memory(data.size)
    cuda_lib.copy_to_gpu(data.ctypes.data, gpu_data, data.size)
    return gpu_data

def gpu_to_cpu(gpu_data, shape):
    print('cuda interface: moving gpu to cpu')
    cpu_data = np.empty(shape, dtype=np.float32)
    cuda_lib.copy_gpu_to_cpu(gpu_data, cpu_data.ctypes.data, cpu_data.size)
    return cpu_data

def relu(gpu_data, size):
    cuda_lib.relu_kernel(gpu_data, size)

def linear(gpu_input, gpu_weights, gpu_bias, m, n, k):
    # TODO: figure out how to manage intermediate memory!!!
    gpu_output = cuda_lib.allocate_gpu_memory(m * n * ctypes.sizeof(ctypes.c_float))
    cuda_lib.linear_kernel(gpu_input, gpu_weights, gpu_bias, gpu_output, m, n, k)
    return gpu_output

def add_matrices(a, b):
    assert a.shape == b.shape, "Matrices must have the same shape."

    rows, cols = a.shape
    c = np.zeros_like(a)
    
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    cuda_lib.add(a_ptr, b_ptr, c_ptr,
            ctypes.c_int(rows),
            ctypes.c_int(cols))

    return c

def matmul_simple(left, right):
    lrows, lcols = left.shape
    rrows, rcols = right.shape
    
    assert lcols == rrows, "Inner dimension of matrices must be equal."

    result = np.zeros((lrows, rcols)).astype(np.float32)

    left_ptr = left.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    right_ptr = right.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    cuda_lib.matmul_simple(left_ptr, right_ptr, result_ptr, lrows, lcols, rrows, rcols)

    return result