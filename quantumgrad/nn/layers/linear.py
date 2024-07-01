import numpy as np
from quantumgrad.nn.module import Module 
from quantumgrad.nn.parameter import Parameter
from quantumgrad import cuda

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32))
        self.bias = Parameter(np.zeros(out_features).astype(np.float32))
        self.add_parameter(self.weight)
        self.add_parameter(self.bias)

    def forward(self, input):
        if self._device == 'cuda':
            # if input is Tensor, send .data
            if hasattr(input, 'data'):
                input = input.data
            return cuda.linear(input, self.weight._data, self.bias._data, self.in_features, self.out_features)
        return np.dot(input, self.weight._data.T) + self.bias._data
    