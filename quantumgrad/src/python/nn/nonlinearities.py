import numpy as np
from ..modules.module import Module
from ..cuda_wrapper import cuda_interface as cuda

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if self._device == 'cuda':
            return cuda.relu(input)
        return np.maximum(input, 0)