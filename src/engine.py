import math
import numpy as np

class Value:
    def __init__(self, data, label=''):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._op = ''
        self._prev = tuple()
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value(data:{self.data})"

    def __rsub__(self, other):
        return Value(other) - self     

    def __radd__(self, other):
        return self.__add__(other)

    def __rtruediv__(self, other): 
        return Value(other) / self
    
    def __add__(self, other):
        other = Value(other, 'other') if not isinstance(other, Value) else other
        out = Value(self.data + other.data, label='add')
        out.ops = '+'
        out._prev = (self, other)

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = Value(other, 'other') if not isinstance(other, Value) else other
        out = Value(self.data - other.data, label='sub')
        out.ops = '-'
        out._prev = (self, other)

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        other = Value(other, 'other') if not isinstance(other, Value) else other
        out = Value(self.data * other.data, label='mul')
        out.ops = '*'
        out._prev = (self, other)
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = Value(other, 'other') if not isinstance(other, Value) else other
        out = Value(self.data / other.data, label='div')
        out.ops = '/'
        out._prev = (self, other)
        
        def _backward():
            self.grad += (1/other.data) * out.grad
            other.grad += -(self.data/(other.data)**2) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        exp = math.exp(self.data)
        exn = math.exp(-self.data)
        tanhx = (exp - exn) / (exp + exn)
        out = Value(tanhx, label=f'tanh({self.label})')
        out.ops = 'tanh'
        out._prev = (self, )
        def _backward():
            self.grad += (1 - (out.data)**2) * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = Value(other, 'other') if not isinstance(other, Value) else other
        out = Value(self.data ** other.data, label='pow')
        out.ops = '**'
        out._prev = (self, other)
        
        def _backward():
            if self.data != 0:
                self.grad += other.data * (self.data)**(other.data - 1.0) * out.grad
            if self.data > 0:
                other.grad +=  out.data * (np.log(self.data)) * out.grad
            if self.data == 0:
                self.grad += 1.0
        out._backward = _backward
        return out


    def backward(self):
        tapolo = []
        seen = set()
        def topo(v):
            if v not in seen:
                seen.add(v)
                for node in v._prev:
                    topo(node)
                tapolo.append(v)
        topo(self)

        self.grad = 1.0
        for node in reversed(tapolo):
            node._backward()