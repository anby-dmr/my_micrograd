
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _child=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_child)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self, ), f"**{other}")

        def _backward():
            self.grad += (other * self.data **(other - 1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self, ), f"ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Get topological graph
        topo = []
        visited = set()

        self.grad = 1.
        def build_topo(u: Value):
            if u not in visited:
                visited.add(u)
                for v in u._prev:
                    build_topo(v)
                topo.append(u)
        build_topo(self)

        # Backprop
        for u in reversed(topo):
            u._backward()

    # Use the above basic arithmetics to build the following.
    def __neg__(self): # -self
        return self * (-1)

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**(-1)

    def __rtruediv__(self, other): # other / self
        return other * self**(-1)

    def __repr__(self):
        return f"Value: {self.data}, grad: {self.grad}"