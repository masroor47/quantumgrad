import numpy as np
from ..cuda_wrapper import cuda_interface as cuda

class Module:
    def __init__(self):
        self._parameters = []
        self._device = 'cpu'

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def to(self, device):
        if self._device == device:
            return self
        for param in self._parameters:
            if device == 'cuda':
                param._data = cuda.cpu_to_gpu(param._data)
            else:
                param._data = cuda.gpu_to_cpu(param._data)
        self._device = device
        return self

    def parameters(self):
        return self._parameters
    
    def add_parameter(self, param):
        self._parameters.append(param)
    
    def forward(self, *inputs):
        raise NotImplementedError
    

class Parameter:
    def __init__(self, data):
        self._data = data

    def to(self, device):
        if device == 'cuda':
            self._data = cuda.cpu_to_gpu(self._data)
        else:
            self._data = cuda.gpu_to_cpu(self._data)
        return self