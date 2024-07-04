import os
import ctypes
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
util_lib_path = os.path.join(current_dir, '../../lib/libcudautils.so')
kernel_lib_path = os.path.join(current_dir, '../../lib/libcudakernels.so')

util_lib = ctypes.cdll.LoadLibrary(util_lib_path)
kernel_lib = ctypes.cdll.LoadLibrary(kernel_lib_path)

kernel_lib.add.argtypes = [ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int]

kernel_lib.matmul_simple.argtypes = [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int]

kernel_lib.multiply_add.argtypes = [ctypes.c_void_p, 
                                    ctypes.c_void_p, 
                                    ctypes.c_void_p, 
                                    ctypes.c_void_p, 
                                    ctypes.c_int, 
                                    ctypes.c_int]

util_lib.allocate_gpu_memory.restype = ctypes.c_void_p
util_lib.allocate_gpu_memory.argtypes = [ctypes.c_size_t]

util_lib.free_gpu_memory.restype = None
util_lib.free_gpu_memory.argtypes = [ctypes.c_void_p]

util_lib.copy_cpu_to_gpu.restype = None
util_lib.copy_cpu_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

util_lib.copy_gpu_to_cpu.restype = None
util_lib.copy_gpu_to_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


def cpu_to_gpu(data):
    # print('cuda wrapper: moving cpu to gpu')
    total_elements = data.size
    total_size_in_bytes = total_elements * np.float32().nbytes
    gpu_data = util_lib.allocate_gpu_memory(total_size_in_bytes)
    util_lib.copy_cpu_to_gpu(data.ctypes.data, gpu_data, total_size_in_bytes)
    return gpu_data

def gpu_to_cpu(gpu_data, shape):
    # print('cuda wrapper: moving gpu to cpu')
    cpu_data = np.empty(shape, dtype=np.float32)
    total_elements = np.prod(shape)
    total_size_in_bytes = total_elements * np.float32().nbytes
    util_lib.copy_gpu_to_cpu(gpu_data, cpu_data.ctypes.data, total_size_in_bytes)
    return cpu_data

def free_gpu_memory(gpu_data):
    util_lib.free_gpu_memory(gpu_data)

def relu(gpu_data, size):
    kernel_lib.relu_kernel(gpu_data, size)

def linear(gpu_input, gpu_weights, gpu_bias, lrows, lcols, rrows, rcols):
    '''
    Multiplies a matrix by vector and adds a bias.
    Input, weights and bias are all pointers to GPU memory.
    '''
    # TODO: figure out how to manage intermediate memory!!!
    gpu_output = util_lib.allocate_gpu_memory(lrows * rcols * np.float32().nbytes)
    kernel_lib.multiply_add(gpu_input, gpu_weights, gpu_bias, gpu_output, lrows, lcols, rrows, rcols)
    return gpu_output

def add_matrices(a, b):
    assert a.shape == b.shape, "Matrices must have the same shape."

    rows, cols = a.shape
    c = np.zeros_like(a)
    
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    kernel_lib.add(a_ptr, b_ptr, c_ptr,
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

    kernel_lib.matmul_simple(left_ptr, right_ptr, result_ptr, lrows, lcols, rrows, rcols)

    return result