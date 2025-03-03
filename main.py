import numpy as np
from tensor import Tensor
from mlp import SimpleMLP
from util import *

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
z = add(mul(x, y), pow(y, 2))  # x*y + y^2
z.backward()  # Compute gradients

print("z:", z.data)
# Expected: [1*4 + 4^2, 2*5 + 5^2, 3*6 + 6^2] = [20.0, 35.0, 54.0]
print("dz/dx:", x.grad)  # Expected: [4.0, 5.0, 6.0]
print("dz/dy:",
      y.grad)  # Expected: [1 + 2*4, 2 + 2*5, 3 + 2*6] = [9.0, 12.0, 15.0]

x.grad = None  # Reset gradients
y.grad = None  # Reset gradients

# Multilayer Perceptron
X = np.random.rand(100)  # placeholder inputs
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100) * 0.1
# y = 3x^2 + 2x + 1 with some noise, placeholder inputs
model = SimpleMLP(input_size=1, hidden_size=5, output_size=1)
model.train(X, y, learning_rate=0.01, epochs=1000)
X_test = np.array([0.1, 0.5, 0.9])
predictions = model.predict(X_test)

for x, pred in zip(X_test, predictions):
    print(
        f"Input: {x:.2f}, Predicted: {pred:.4f}, Expected: {3*x**2 + 2*x + 1:.4f}"
    )
print(f"W1: {model.W1.data}")
print(f"b1: {model.b1.data}")
print(f"W2: {model.W2.data}")
print(f"b2: {model.b2.data}")
