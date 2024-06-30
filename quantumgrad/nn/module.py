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
            return 

        # print(f'\nModule; before moving to {device}\n')
        # for param in self._parameters:
        #     print(f"param.device: {param.device}; {param}")
        
        # print(f"weight device: {self.weight.device}; {self.weight}")
        # print(f"bias device: {self.bias.device}; {self.bias}")
        # print()
        
        for i, param in enumerate(self._parameters):
            new_param = param.to(device)
            self._parameters[i] = new_param
            # Update corresponding attribute if it exists
            for key, value in self.__dict__.items():
                if value is param:
                    # print(f'key: {key}; value: {value}; \nparam: {param}; \nnew_param: {new_param}')
                    setattr(self, key, new_param)

            # print()
        
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                # print("moving submodules to device")
                # print(f"{name = };  {module = }")
                module.to(device)
                # setattr(self, name, module.to(device))

        # print device of every parameter
        # print()
        # for param in self._parameters:
        #     print(f"param.device: {param.device}; {param}")
        
        # print(f"weight device: {self.weight.device}; {self.weight}")
        # print(f"bias device: {self.bias.device}; {self.bias}")
        # print()


        self._device = device
        # print(f"Moved module to {self._device}")
        # return self

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