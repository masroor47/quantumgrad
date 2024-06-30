import numpy as np
from quantumgrad.nn.module import Module 
from quantumgrad.nn.parameter import Parameter
from quantumgrad import cuda

class Linear(Module):
    def __init__(self, in_features, out_features):
        # print('Creating Linear layer')
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # print('in Linear, about to create weight and bias parameters')
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32))
        self.bias = Parameter(np.zeros(out_features).astype(np.float32))
        self.add_parameter(self.weight)
        self.add_parameter(self.bias)

    # def to(self, device):
    #     super().to(device)
    #     self.weight = self.weight.to(device)
    #     self.bias = self.bias.to(device)
    #     return self

    def forward(self, input):
        if self._device == 'cuda':
            return cuda.linear(input, self.weight._data, self.bias._data)
        return np.dot(input, self.weight._data.T) + self.bias._data
    