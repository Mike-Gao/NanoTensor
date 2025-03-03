from tensor import Tensor
import numpy as np


def add(tensor1, tensor2):
    data = tensor1.data + tensor2.data
    out = Tensor(data,
                 requires_grad=(tensor1.requires_grad
                                or tensor2.requires_grad))

    def _backward():
        if tensor1.requires_grad:
            if tensor1.grad is None:
                tensor1.grad = np.zeros_like(tensor1.data)
            tensor1.grad += out.grad
        if tensor2.requires_grad:
            if tensor2.grad is None:
                tensor2.grad = np.zeros_like(tensor2.data)
            tensor2.grad += out.grad

    out._backward = _backward
    out._prev = {tensor1, tensor2}
    return out


def mul(tensor1, tensor2):
    data = tensor1.data * tensor2.data
    out = Tensor(data,
                 requires_grad=(tensor1.requires_grad
                                or tensor2.requires_grad))

    def _backward():
        if tensor1.requires_grad:
            if tensor1.grad is None:
                tensor1.grad = np.zeros_like(tensor1.data)
            tensor1.grad += out.grad * tensor2.data
        if tensor2.requires_grad:
            if tensor2.grad is None:
                tensor2.grad = np.zeros_like(tensor2.data)
            tensor2.grad += out.grad * tensor1.data

    out._backward = _backward
    out._prev = {tensor1, tensor2}
    return out


def sub(tensor1, tensor2):
    data = tensor1.data - tensor2.data
    out = Tensor(data,
                 requires_grad=(tensor1.requires_grad
                                or tensor2.requires_grad))

    def _backward():
        if tensor1.requires_grad:
            if tensor1.grad is None:
                tensor1.grad = np.zeros_like(tensor1.data)
            tensor1.grad += out.grad
        if tensor2.requires_grad:
            if tensor2.grad is None:
                tensor2.grad = np.zeros_like(tensor2.data)
            tensor2.grad -= out.grad

    out._backward = _backward
    out._prev = {tensor1, tensor2}
    return out


def mean(tensor):
    data = tensor.data.mean()
    out = Tensor(data, requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            # Gradient of mean w.r.t. each element is 1 / number of elements
            tensor.grad += out.grad * np.ones_like(
                tensor.data) / tensor.data.size

    out._backward = _backward
    out._prev = {tensor}
    return out


def matmul(tensor1, tensor2):
    data = tensor1.data @ tensor2.data
    out = Tensor(data,
                 requires_grad=(tensor1.requires_grad
                                or tensor2.requires_grad))

    def _backward():
        if tensor1.requires_grad:
            if tensor1.grad is None:
                tensor1.grad = np.zeros_like(tensor1.data)
            tensor1.grad += out.grad @ tensor2.data.T
        if tensor2.requires_grad:
            if tensor2.grad is None:
                tensor2.grad = np.zeros_like(tensor2.data)
            tensor2.grad += tensor1.data.T @ out.grad

    out._backward = _backward
    out._prev = {tensor1, tensor2}
    return out


def pow(tensor, exponent):
    data = tensor.data**exponent
    out = Tensor(data, requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            tensor.grad += out.grad * exponent * (tensor.data**(exponent - 1))

    out._backward = _backward
    out._prev = {tensor}
    return out


def sgd(params, grads, learning_rate):
    """
    Performs Stochastic Gradient Descent (SGD) optimization.

    Parameters:
    - params: List of tensors (model parameters) to be updated.
    - grads: List of tensors (gradients of the parameters).
    - learning_rate: The learning rate for optimization.
    """
    for param, grad in zip(params, grads):
        if grad is not None:
            param.data -= learning_rate * grad


def relu(tensor):
    data = np.maximum(0, tensor.data)
    out = Tensor(data, requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            tensor.grad += out.grad * (tensor.data > 0).astype(float)

    out._backward = _backward
    out._prev = {tensor}
    return out