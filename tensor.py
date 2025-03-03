import numpy as np
class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = None
        self._prev = set()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for t in tensor._prev:
                    build_topo(t)
                topo.append(tensor)

        build_topo(self)
        for t in reversed(topo):
            if t._backward:
                t._backward()
