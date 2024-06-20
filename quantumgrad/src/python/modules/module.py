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
        for param in self.parameters():
            param.to(device)
        self._device = device
        return self

    def parameters(self):
        params = self._parameters[:]
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                params.extend(module.parameters())
        return params
    
    def add_parameter(self, param):
        self._parameters.append(param)
    
    def forward(self, *inputs):
        raise NotImplementedError
    

class Parameter:
    def __init__(self, data):
        self._data = data
        self._device = 'cpu'

    def to(self, device):
        if device == 'cuda':
            self._data = cuda.cpu_to_gpu(self._data)
        else:
            self._data = cuda.gpu_to_cpu(self._data)
        self._device = device
        return self
    
    def numel(self):
        if self._device == 'cpu':
            return self._data.size
        else:
            # TODO: count param size if data is on cuda
            print("device is cuda, can't give you the size rn")
            raise NotImplementedError