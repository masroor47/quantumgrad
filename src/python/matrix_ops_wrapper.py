import os
import ctypes
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, '../../lib/matrix_ops.so')
lib = ctypes.cdll.LoadLibrary(lib_path)

lib.add.argtypes = [ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int]

lib.matmul_simple.argtypes = [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int]

def add_matrices(a, b):
    assert a.shape == b.shape, "Matrices must have the same shape."

    rows, cols = a.shape
    c = np.zeros_like(a)
    
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.add(a_ptr, b_ptr, c_ptr,
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

    lib.matmul_simple(left_ptr, right_ptr, result_ptr, lrows, lcols, rrows, rcols)

    return result