

from quantumgrad.src.python.modules.module import Module
from quantumgrad.src.python.modules.layer import Linear
from quantumgrad.src.python.nn.nonlinearities import ReLU

import numpy as np

if __name__ == '__main__':
    linear = Linear(784, 10)
    relu = ReLU()
    output = relu(linear(np.random.randn(1, 784)))
    print(output.shape)
    print(output)