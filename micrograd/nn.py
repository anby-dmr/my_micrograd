import random
from micrograd.engine import Value

class Module:
    # Zero out gradients
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []

    # Return all parameters

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        # Randomly initialize w, and zero out b
        # nonline is True for ReLU, False for linear
        self.w = [Value(random.uniform(-1., 1.)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        # Compute the result of the neuron
        output = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return output.relu() if self.nonlin else output

    def parameters(self):
        # Return the parameters of the neuron (as a list of Values)
        return self.w + [self.b]

    def __repr__(self):
        # Nonlinearity and number of inputs
        return ("ReLU" if self.nonlin else "Linear") + f"Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        # Create a list of neurons with nin inputs and nout outputs
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        ouput = [n(x) for n in self.neurons]
        return ouput[0] if len(ouput)==1 else ouput

    def parameters(self):
        # Parameters of all neurons in the layer (as a list)
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        # Return a list, each element is a neuron's string representation
        return f"Layer of [{", ".join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
        # nin is the number of inputs, nouts is a list of layer sizes
        # Create a list of layers. The last layer should not have a nonlinearity.
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i + 1 < len(sz) - 1)) for i in range(len(sz) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def __repr__(self):
        return f"MLP of [{", ".join(str(l) for l in self.layers)}]"
