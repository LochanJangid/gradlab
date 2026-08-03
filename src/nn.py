class Neuron:
    def __init__(self, nin, nonlin):
        self.w = [Value(np.random.uniform(-1,1), label='w') for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1), label='b')
        self.nonlin = nonlin

    def __call__(self, x):
        out = sum(([wi*xi for wi, xi in zip(self.w, x)]), self.b)
        return out.tanh() if self.nonlin else out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, nonlin):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        nn = [nin] + nouts
        self.layers = [Layer(nn[i], nn[i+1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]