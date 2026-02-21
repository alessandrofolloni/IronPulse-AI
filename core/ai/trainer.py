import numpy as np
from .engine import PulseMindNet
import datetime

class PulseMindTrainer:
    def __init__(self, model):
        self.model = model
        self.learning_rate = 0.01

    def train_step(self, X, y_true):
        # Forward pass
        y_pred = self.model.forward(X)
        
        # Loss (Cross-Entropy for softmax)
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        
        # Backward pass
        # Output layer gradient
        dz = (y_pred - y_true) / m
        
        # Gradient for last weights/biases
        dw = np.dot(self.model.activations[-2].T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        
        # Update weights/biases (simple SGD)
        self.model.weights[-1] -= self.learning_rate * dw
        self.model.biases[-1] -= self.learning_rate * db
        
        # Propagate back to hidden layers
        curr_dz = dz
        for i in range(len(self.model.weights) - 2, -1, -1):
            da = np.dot(curr_dz, self.model.weights[i+1].T)
            # ReLU derivative
            dz_hidden = da * (self.model.activations[i+1] > 0)
            
            dw_hidden = np.dot(self.model.activations[i].T, dz_hidden)
            db_hidden = np.sum(dz_hidden, axis=0, keepdims=True)
            
            self.model.weights[i] -= self.learning_rate * dw_hidden
            self.model.biases[i] -= self.learning_rate * db_hidden
            curr_dz = dz_hidden
            
        return loss

    def train(self, X, y, epochs=100):
        losses = []
        for epoch in range(epochs):
            loss = self.train_step(X, y)
            losses.append(loss)
        return losses

def prepare_workout_data(sessions, exercises):
    """
    Dummy data preparation for initial AI build.
    In a real scenario, this would transform Django ORM data to Numpy arrays.
    """
    # Features: [Muscle group index (10), Hour of day (1), Day of week (7)] = 18 inputs
    # Targets: Exercise index (N)
    num_samples = max(10, len(sessions))
    X = np.random.rand(num_samples, 18)
    y = np.zeros((num_samples, len(exercises)))
    for i in range(num_samples):
        y[i, np.random.randint(0, len(exercises))] = 1
        
    return X, y
