import numpy as np
import quantumgrad.cuda as cuda

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