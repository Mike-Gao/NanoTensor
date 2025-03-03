from tensor import Tensor
import numpy as np
from util import *


class SimpleMLP:

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases for input to hidden layer
        # Weights of the first layer
        self.W1 = Tensor(np.random.randn(input_size, hidden_size) * 0.01,
                         requires_grad=True)
        self.b1 = Tensor(np.zeros((1, hidden_size)),
                         requires_grad=True)  # bias of the first layer
        # Initialize weights and biases for hidden to output layer
        self.W2 = Tensor(np.random.randn(hidden_size, output_size) * 0.01,
                         requires_grad=True)
        self.b2 = Tensor(np.zeros((1, output_size)), requires_grad=True)

    def forward(self, x):
        self.X = x
        self.Z1 = add(matmul(x, self.W1),
                      self.b1)  # Output of the first layer (before activation)
        self.A1 = relu(
            self.Z1
        )  # Output of the first layer after applying ReLU activation
        self.Z2 = add(matmul(self.A1, self.W2),
                      self.b2)  # Output of the second layer (final prediction)
        return self.Z2

    def compute_loss(self, y_pred, y_true):
        # Mean Squared Error Loss
        return mean(pow(sub(y_pred, y_true), 2))

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0

            for i in range(len(X)):
                x = Tensor([[X[i]]])
                y_true = Tensor([[y[i]]])

                # Forward pass
                y_pred = self.forward(x)

                # Compute loss
                loss = self.compute_loss(y_pred, y_true)
                total_loss += loss.data

                # Backward pass
                loss.backward()

                # Update parameters
                params = [self.W1, self.b1, self.W2, self.b2]
                grads = [
                    self.W1.grad, self.b1.grad, self.W2.grad, self.b2.grad
                ]
                sgd(params, grads, learning_rate)
                # Reset gradients
                for param in params:
                    param.grad = None

            avg_loss = total_loss / len(X)  # Print average loss for this epoch
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    def predict(self, X):
        X = X.reshape(-1, 1)
        predictions = []
        for x in X:
            x_tensor = Tensor([[x]])
            pred = self.forward(x_tensor)
            # Extract scalar from tensor for printing
            predictions.append(float(pred.data[0, 0]))
        return predictions
