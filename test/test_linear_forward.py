import unittest
import numpy as np
from quantumgrad import Tensor, nn

class TestLinearForward(unittest.TestCase):
    '''
    Test the forward pass of the Linear layer.
    One dimentional input as well as batch input is tested.
    '''
    def setUp(self):
        np.random.seed(42)
        self.input_dim = 200
        self.output_dim = 100
        self.batch_size = 16

    def test_linear_forward_cpu(self):
        linear_layer = nn.Linear(self.input_dim, self.output_dim)
        x = np.random.rand(self.input_dim).astype(np.float32)
        y = linear_layer.forward(x)
        self.assertEqual(y.shape, (self.output_dim,))
        np_result = np.dot(x, linear_layer.weight._data.T) + linear_layer.bias._data
        np.testing.assert_allclose(y, np_result, rtol=1e-5, atol=1e-5)
    
    def test_linear_forward_batch_input_cpu(self):
        linear_layer = nn.Linear(self.input_dim, self.output_dim)
        x = np.random.rand(self.input_dim, self.batch_size).astype(np.float32)
        y = linear_layer.forward(x)
        self.assertEqual(y.shape, (self.output_dim, self.batch_size))
        np_result = np.dot(linear_layer.weight._data, x) + linear_layer.bias._data[:, np.newaxis]
        np.testing.assert_allclose(y, np_result, rtol=1e-5, atol=1e-5)

    def test_linear_forward_cuda(self):
        linear_layer = nn.Linear(self.input_dim, self.output_dim)
        cpu_weights_data = linear_layer.weight.data.copy()
        cpu_nonzeor_bias = np.random.rand(self.output_dim).astype(np.float32)
        linear_layer.bias._data = cpu_nonzeor_bias

        linear_layer.to('cuda')
        self.assertEqual(linear_layer.weight.device, 'cuda')

        np_input = np.random.rand(self.input_dim).astype(np.float32)
        tensor_input = Tensor(np_input).to('cuda')

        tensor_output = linear_layer(tensor_input).to('cpu')
        np_output = np.dot(np_input, cpu_weights_data.T) + cpu_nonzeor_bias

        np.testing.assert_allclose(tensor_output.data[:, 0], np_output, rtol=1e-5, atol=1e-5)
    
    def test_linear_forward_batch_input_cuda(self):
        linear_layer = nn.Linear(self.input_dim, self.output_dim)
        cpu_weights_data = linear_layer.weight.data.copy()
        cpu_nonzeor_bias = np.random.rand(self.output_dim).astype(np.float32)
        linear_layer.bias._data = cpu_nonzeor_bias

        linear_layer.to('cuda')
        self.assertEqual(linear_layer.weight.device, 'cuda')

        np_input = np.random.rand(self.input_dim, self.batch_size).astype(np.float32)
        tensor_input = Tensor(np_input).to('cuda')

        tensor_output = linear_layer(tensor_input).to('cpu')
        np_output = np.dot(cpu_weights_data, np_input) + cpu_nonzeor_bias[:, np.newaxis]

        np.testing.assert_allclose(tensor_output.data, np_output, rtol=1e-5, atol=1e-5)
    
    def test_linear_forward_relu_cpu(self):
        linear_layer = nn.Linear(self.input_dim, self.output_dim)
        x = np.random.rand(self.input_dim).astype(np.float32)
        y = linear_layer.forward_relu(x)
        self.assertEqual(y.shape, (self.output_dim,))
        np_result = np.maximum(0, np.dot(x, linear_layer.weight._data.T) + linear_layer.bias._data)
        np.testing.assert_allclose(y, np_result, rtol=1e-5, atol=1e-5)

    def test_linear_forward_relu_batch_input_cpu(self):
        linear_layer = nn.Linear(self.input_dim, self.output_dim)
        x = np.random.rand(self.input_dim, self.batch_size).astype(np.float32)
        y = linear_layer.forward_relu(x)
        self.assertEqual(y.shape, (self.output_dim, self.batch_size))
        np_result = np.maximum(0, np.dot(linear_layer.weight._data, x) + linear_layer.bias._data[:, np.newaxis])
        np.testing.assert_allclose(y, np_result, rtol=1e-5, atol=1e-5)

    @unittest.skip("Skip CUDA tests for now")
    def test_linear_forward_relu_cuda(self):
        linear_layer = nn.Linear(self.input_dim, self.output_dim)
        cpu_weights_data = linear_layer.weight.data.copy()
        cpu_nonzeor_bias = np.random.rand(self.output_dim).astype(np.float32)
        linear_layer.bias._data = cpu_nonzeor_bias

        linear_layer.to('cuda')
        self.assertEqual(linear_layer.weight.device, 'cuda')

        np_input = np.random.rand(self.input_dim).astype(np.float32)
        tensor_input = Tensor(np_input).to('cuda')

        tensor_output = linear_layer.forward_relu(tensor_input).to('cpu')
        np_output = np.maximum(0, np.dot(np_input, cpu_weights_data.T) + cpu_nonzeor_bias)

        np.testing.assert_allclose(tensor_output.data[:, 0], np_output, rtol=1e-5, atol=1e-5)

    @unittest.skip("Skip CUDA tests for now")
    def test_linear_forward_relu_batch_input_cuda(self):
        linear_layer = nn.Linear(self.input_dim, self.output_dim)
        cpu_weights_data = linear_layer.weight.data.copy()
        cpu_nonzeor_bias = np.random.rand(self.output_dim).astype(np.float32)
        linear_layer.bias._data = cpu_nonzeor_bias

        linear_layer.to('cuda')
        self.assertEqual(linear_layer.weight.device, 'cuda')

        np_input = np.random.rand(self.input_dim, self.batch_size).astype(np.float32)
        tensor_input = Tensor(np_input).to('cuda')

        tensor_output = linear_layer.forward_relu(tensor_input).to('cpu')
        np_output = np.maximum(0, np.dot(cpu_weights_data, np_input) + cpu_nonzeor_bias[:, np.newaxis])

        np.testing.assert_allclose(tensor_output.data, np_output, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()