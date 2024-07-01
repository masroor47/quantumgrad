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
            return 

        for i, param in enumerate(self._parameters):
            new_param = param.to(device)
            self._parameters[i] = new_param
            # Update corresponding attribute if it exists (weight, bias)
            for key, value in self.__dict__.items():
                if value is param:
                    setattr(self, key, new_param)
        
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                module.to(device)

        self._device = device

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