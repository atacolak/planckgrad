import math

class Value:
    def __init__(self, data, _prev = (), _op = ''):
        self.data = data
        self.grad = 0.0
        self._backward = None 
        self._prev = _prev
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), _op='-')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return self - other
    
    def  __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, x):
        assert isinstance(x, (int, float)), "Only supporting scalar exponentiation"
        out = Value(self.data ** x, (self, ), _op=f'**{x}')
        def _backward():
            self.grad += (x * self.data**(x-1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1
    
    # def __rdiv__(self, other):
    #     return self * other**-1

    def exp(self): # e**x
        out = Value(math.exp(self.data), (self, ), _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self): # Sigmoid
        x = self.data
        sig = 1 / (1 + math.exp(-x))
        #sig = (1 + (-self).exp())**-1
        out = Value(sig, (self, ), _op='sigmoid')
        def _backward():
            self.grad += (sig * (1 - sig)) * out.grad
        out._backward = _backward
        return out
    
    def silu(self): #Sigmoid Linear Unit (Swish)
        x = self.data
        sig = self.sigmoid().data
        out = Value(x * sig, (self, ), _op='silu')
        def _backward():
            self.grad += ((x * (sig * (1 - sig))) + sig) * out.grad
        out._backward = _backward
        return out

    def backprop(self): #use topological sort for all children and reverse the result
        values = []
        seen = set()
        def build_topo(self):
            stack = [self]
            while stack:
                node = stack.pop()
                if node not in seen:
                    seen.add(node)
                    stack.extend(node._prev)  # Add parents to stack
                    values.append(node)  # Add self AFTER parents
        build_topo(self)

        for v in values:
            v.grad = 0.0 # Reset gradients from previous backpropagation step
        self.grad = 1.0 # Seed gradient for this node (starting point for backprop)

        [v._backward() for v in values if v._backward is not None] #backprop from the last node (BFS)