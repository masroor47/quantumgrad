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
    
    def test_linear_forward_batch_input_cpu(self):
        input_dim = 200
        output_dim = 100
        batch_size = 16
        linear = nn.Linear(input_dim, output_dim)
        x = np.random.rand(input_dim, batch_size).astype(np.float32)
        y = linear.forward(x)
        self.assertEqual(y.shape, (output_dim, batch_size))
        np_result = np.dot(linear.weight._data, x) + linear.bias._data[:, np.newaxis]
        np.testing.assert_allclose(y, np_result, rtol=1e-5, atol=1e-5)

    def test_linear_forward_cuda(self):
        input_dim = 200
        output_dim = 100
        linear_layer = nn.Linear(input_dim, output_dim)
        cpu_weights_data = linear_layer.weight.data.copy()
        cpu_nonzeor_bias = np.random.rand(output_dim).astype(np.float32)
        linear_layer.bias.data = cpu_nonzeor_bias
        linear_layer.to('cuda')
        self.assertEqual(linear_layer.weight.device, 'cuda')
        np_input = np.random.rand(input_dim).astype(np.float32)
        tensor_input = Tensor(np_input).to('cuda')
        tensor_output = linear_layer(tensor_input).to('cpu')
        np_output = np.dot(np_input, cpu_weights_data.T) + cpu_nonzeor_bias
        np.testing.assert_allclose(tensor_output.data[:, 0], np_output, rtol=1e-5, atol=1e-5)
    
    def test_linear_forward_batch_input_cuda(self):
        input_dim = 200
        output_dim = 100
        batch_size = 16
        linear_layer = nn.Linear(input_dim, output_dim)
        cpu_weights_data = linear_layer.weight.data.copy()
        cpu_nonzeor_bias = np.random.rand(output_dim).astype(np.float32)
        linear_layer.bias.data = cpu_nonzeor_bias
        linear_layer.to('cuda')
        self.assertEqual(linear_layer.weight.device, 'cuda')
        np_input = np.random.rand(input_dim, batch_size).astype(np.float32)
        tensor_input = Tensor(np_input).to('cuda')
        tensor_output = linear_layer(tensor_input).to('cpu')
        cpu_bias_data_expanded = np.tile(cpu_nonzeor_bias[:, np.newaxis], (1, batch_size))
        print(f"cpu_bias_data_expanded: {cpu_bias_data_expanded.shape}")
        np_output = np.dot(cpu_weights_data, np_input) + cpu_bias_data_expanded
        print(f"np_output: {np_output.shape}")
        print(f"tensor_output: {tensor_output.shape}")
        np.testing.assert_allclose(tensor_output.data, np_output, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()