import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary('./lib/matrix_ops.so')

lib.matrix_add.argtypes = [ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.c_int,
                           ctypes.c_int]

def add_matrices(a, b):
    assert a.shape == b.shape, "Matrices must have the same shape"

    rows, cols = a.shape
    c = np.zeros_like(a)
    
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.matrix_add(a_ptr, b_ptr, c_ptr,
                   ctypes.c_int(rows),
                   ctypes.c_int(cols))

    return c