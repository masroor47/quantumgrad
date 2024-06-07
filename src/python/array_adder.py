import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary('./lib/matrix_ops.so')

lib.array_add.argtypes = [ctypes.POINTER(ctypes.c_float),
                          ctypes.POINTER(ctypes.c_float),
                          ctypes.POINTER(ctypes.c_float),
                          ctypes.c_int]

def add_arrays(a, b):
    assert a.shape == b.shape, "arrays must have the same length"

