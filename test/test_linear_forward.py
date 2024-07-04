import unittest
import numpy as np
from quantumgrad import Tensor, nn

class TestLinearForward(unittest.TestCase):
    '''
    Test the forward pass of the Linear layer.
    '''
    def test_linear_forward_cpu(self):
        input_dim = 200
        output_dim = 100
        linear = nn.Linear(input_dim, output_dim)
        x = np.random.rand(input_dim).astype(np.float32)
        y = linear.forward(x)
        self.assertEqual(y.shape, (output_dim,))
        np_result = np.dot(x, linear.weight._data.T) + linear.bias._data
        np.testing.assert_allclose(y, np_result, rtol=1e-5, atol=1e-5)

    def test_linear_forward_cuda(self):
        input_dim = 200
        output_dim = 100
        linear_layer = nn.Linear(input_dim, output_dim)
        cpu_weights_data = linear_layer.weight.data.copy()
        cpu_bias_data = linear_layer.bias.data.copy()
        linear_layer.to('cuda')
        self.assertEqual(linear_layer.weight.device, 'cuda')
        np_input = np.random.rand(input_dim).astype(np.float32)
        tensor_input = Tensor(np_input).to('cuda')
        tensor_output = linear_layer.forward(tensor_input).to('cpu')
        np_output = np.dot(np_input, cpu_weights_data.T) + cpu_bias_data
        np.testing.assert_allclose(tensor_output.data, np_output, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()