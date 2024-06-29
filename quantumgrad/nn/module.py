import numpy as np
import quantumgrad.cuda as cuda

class Module:
    def __init__(self):
        # print('Creating Module')
        self._parameters = []
        self._device = 'cpu'

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def to(self, device):
        if self._device == device:
            return self
        
        for i, param in enumerate(self._parameters):
            self._parameters[i] = param.to(device)
        
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                setattr(self, name, module.to(device))

        # print device of every parameter
        print()
        for param in self._parameters:
            print(f"param.device: {param.device}")
        print()


        self._device = device
        print(f"Moved module to {self._device}")
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