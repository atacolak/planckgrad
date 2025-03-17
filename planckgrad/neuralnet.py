from value import Value
import random

class Neuron:
    def __init__(self, count: int):
        # act(w * x + b)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(count)]
        self.b = Value(random.uniform(-1, 1))

    def act(self, xs: list): 
        return (sum([wi * xi for wi, xi in zip(self.w, xs)]) + self.b).silu()

    def __repr__(self): # neat printout
        return '\n'.join([f"    Weight {i+1}: {self.w[i].data:12.7f}" for i in range(len(self.w))]) + f"\n   \tBias: {self.b.data:14.7f}"
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __repr__(self):
        return '\n'.join([f"  Neuron {i+1}\n{self.neurons[i]}" for i in range(len(self.neurons))])

    def act(self, xs: list):
        return [n.act(xs) for n in self.neurons]
    
class MLP:
    def __init__(self, layers: list): # [1, 3, 5, 3, 1] -> 1,3;3,5
        self.layers = [Layer(layers[i], layers[i+1]) for i in range(len(layers)-1)]

    def __repr__(self):
        return '\n'.join([f"Layer {i+1}\n{self.layers[i]}" for i in range(len(self.layers))])
    
    def all_values(self):
        values = []
        for layer in self.layers:
            for neuron in layer.neurons:
                for wt in neuron.w:
                    values.append(wt)
                values.append(neuron.b)
        return values
    
    def __call__(self, xs: list):
        for layer in self.layers:
            pred = layer.act(xs)
        return pred
    
    def train(self, input, desired_output, epochs: int, step_size: int):
        input_size = len(self.layers[0].neurons[0].w)
        for x in input:
            assert len(x) == input_size, \
                f"Expected input size {input_size}, got {len(x)}"  
        for i in range(epochs):
            preds = [self(x) for x in input]
            n_preds = []
            for sublist in preds:
                for pred in sublist:
                    n_preds.append(pred)
            loss = sum([(ground_truth - pred)**2 for ground_truth, pred in zip(desired_output, n_preds)])
            loss.backprop()
            for val in self.all_values():
                val.data += -step_size * val.grad
        print(f"Trained for {i+1} epochs. loss: {loss.data:.5f}")