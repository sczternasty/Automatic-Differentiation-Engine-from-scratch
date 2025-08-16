"""
Neural Network Modules

This module provides neural network building blocks including layers,
activation functions, and complete network architectures. All components
are built on top of the autograd engine for automatic differentiation.
"""

import random
from typing import List, Union, Any
from .autograd import Var


class Module:
    """
    Base class for all neural network modules.
    
    This class provides common functionality for parameter management,
    gradient zeroing, and module composition.
    """
    
    def __init__(self):
        self.sequential = []
    
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self) -> List[Var]:
        return []
    
    def __call__(self, x: Any) -> Any:
        raise NotImplementedError("Subclasses must implement __call__")


class Perceptron(Module):
    """
    A single neuron (perceptron) with learnable weights and bias.
    
    This is the basic building block of neural networks, implementing
    a weighted sum followed by an optional activation function.
    """
    
    def __init__(self, nin: int):
        super().__init__()
        self.weight = [Var(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Var(random.uniform(-1, 1))
    
    def __call__(self, x: List[Union[Var, float]]) -> Var:
        x_vars = [xi if isinstance(xi, Var) else Var(float(xi)) for xi in x]
        out = sum((wi * xi for wi, xi in zip(self.weight, x_vars)), self.bias)
        return out
    
    def parameters(self) -> List[Var]:
        return self.weight + [self.bias]
    
    def __repr__(self) -> str:
        return f"Perceptron({len(self.weight)})"


class Linear(Module):
    """
    A fully connected (linear) layer.
    
    This layer applies a linear transformation to the input data,
    transforming from in_features to out_features dimensions.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.neurons = [Perceptron(in_features) for _ in range(out_features)]
    
    def __call__(self, x: List[Union[Var, float]]) -> Union[Var, List[Var]]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self) -> List[Var]:
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self) -> str:
        return f"Linear({len(self.neurons[0].weight)} -> {len(self.neurons)})"


class ReLU(Module):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    ReLU applies the function f(x) = max(0, x), which introduces
    non-linearity while being computationally efficient.
    """
    
    def __call__(self, x: Union[Var, List[Var]]) -> Union[Var, List[Var]]:
        if isinstance(x, list):
            return [i.relu() for i in x]
        else:
            return x.relu()
    
    def __repr__(self) -> str:
        return "ReLU"
    
    def parameters(self) -> List[Var]:
        return []


class Tanh(Module):
    """
    Hyperbolic tangent activation function.
    
    Tanh applies the function f(x) = (e^x - e^(-x)) / (e^x + e^(-x)),
    which maps inputs to the range [-1, 1] and is useful for
    normalizing outputs.
    """
    
    def __call__(self, x: Union[Var, List[Var]]) -> Union[Var, List[Var]]:
        if isinstance(x, list):
            return [i.tanh() for i in x]
        else:
            return x.tanh()
    
    def __repr__(self) -> str:
        return "Tanh"
    
    def parameters(self) -> List[Var]:
        return []


class MLP(Module):
    """
    Multi-Layer Perceptron (MLP) neural network.
    
    This is a feedforward neural network with multiple layers,
    typically used for classification and regression tasks.
    """
    
    def __init__(self, layer_sizes: List[int] = None):
        super().__init__()
        
        if layer_sizes is None:
            layer_sizes = [3, 4, 4, 1]
        
        self.sequential = []
        for i in range(len(layer_sizes) - 1):
            self.sequential.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.sequential.append(Tanh())
    
    def __call__(self, x: Union[List[List[float]], List[float], float]) -> List[Var]:
        if isinstance(x, list):
            if isinstance(x[0], list):
                x = [list(map(Var, r)) for r in x]
            else:
                x = [list(map(Var, x))]
        else:
            x = [Var(x)]
        
        preds = []
        for i in x:
            for layer in self.sequential:
                i = layer(i)
            preds.append(i)
        
        return preds
    
    def parameters(self) -> List[Var]:
        return [p for layer in self.sequential for p in layer.parameters()]
    
    def __repr__(self) -> str:
        return f"MLP({[str(layer) for layer in self.sequential]})"


class Moon_MLP(Module):
    """
    Specialized MLP for the moons classification dataset.
    
    This network is specifically designed for binary classification
    on 2D data with 2 hidden layers of 16 neurons each.
    """
    
    def __init__(self):
        super().__init__()
        self.sequential = [
            Linear(2, 16),
            ReLU(),
            Linear(16, 16),
            ReLU(),
            Linear(16, 1)
        ]
    
    def __call__(self, x: Union[List[List[float]], List[float], float]) -> List[Var]:
        if isinstance(x, list):
            if isinstance(x[0], list):
                x = [list(map(Var, r)) for r in x]
            else:
                x = [list(map(Var, x))]
        else:
            x = [Var(x)]
        
        preds = []
        for i in x:
            for layer in self.sequential:
                i = layer(i)
            preds.append(i)
        
        return preds
    
    def __repr__(self) -> str:
        return f"Moon_MLP([{', '.join(str(layer) for layer in self.sequential)}])"
    
    def parameters(self) -> List[Var]:
        return [p for layer in self.sequential for p in layer.parameters()]


class CH_MLP(Module):
    """
    Specialized MLP for the California Housing regression dataset.
    
    This network is designed for regression on tabular data with
    13 input features and 2 hidden layers of 64 neurons each.
    """
    
    def __init__(self):
        super().__init__()
        self.sequential = [
            Linear(13, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 1)
        ]
    
    def __call__(self, x: Union[List[List[float]], List[float], float]) -> List[Var]:
        if isinstance(x, list):
            if isinstance(x[0], list):
                x = [list(map(Var, r)) for r in x]
            else:
                x = [list(map(Var, x))]
        else:
            x = [Var(x)]
        
        preds = []
        for i in x:
            for layer in self.sequential:
                i = layer(i)
            preds.append(i)
        
        return preds
    
    def __repr__(self) -> str:
        return f"CH_MLP([{', '.join(str(layer) for layer in self.sequential)}])"
    
    def parameters(self) -> List[Var]:
        return [p for layer in self.sequential for p in layer.parameters()]

