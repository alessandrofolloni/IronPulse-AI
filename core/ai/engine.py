import numpy as np
import json
import os

class PulseMindNet:
    """
    A custom Neural Network implemented from scratch using NumPy.
    Designed for Workout Optimization and Planning.
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights and biases (He initialization)
        self.weights = []
        self.biases = []
        
        # Input to first hidden
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2./input_size))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        # Hidden to hidden
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * np.sqrt(2./hidden_sizes[i]))
            self.biases.append(np.zeros((1, hidden_sizes[i+1])))
            
        # Last hidden to output
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2./hidden_sizes[-1]))
        self.biases.append(np.zeros((1, output_size)))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        curr_input = X
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(curr_input, self.weights[i]) + self.biases[i]
            curr_input = self.relu(z)
            self.activations.append(curr_input)
            
        # Output layer
        z_out = np.dot(curr_input, self.weights[-1]) + self.biases[-1]
        out = self.softmax(z_out)
        self.activations.append(out)
        return out

    def get_parameters(self):
        params = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'config': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'output_size': self.output_size
            }
        }
        return params

    def load_parameters(self, params):
        self.weights = [np.array(w) for w in params['weights']]
        self.biases = [np.array(b) for b in params['biases']]

    @classmethod
    def from_dict(cls, params):
        cfg = params['config']
        net = cls(cfg['input_size'], cfg['hidden_sizes'], cfg['output_size'])
        net.load_parameters(params)
        return net
