"""
Automatic Differentiation Engine (Autograd)

This module implements a complete automatic differentiation system that enables
training neural networks using backpropagation. It provides a Variable class
that tracks computational graphs and automatically computes gradients.
"""

import math
from typing import Set, Tuple, Union, Optional


class Var:
    """
    A variable that supports automatic differentiation.
    
    This class implements the core of the autograd engine, tracking
    computational graphs and computing gradients automatically.
    
    Attributes:
        x: The current value of the variable
        grad: The gradient with respect to this variable
        _backward: Function to compute gradients during backward pass
        previous: Set of child variables in the computational graph
        operation: String representation of the operation that created this variable
        label: Human-readable label for visualization and debugging
    """
    
    def __init__(self, x: float, children: Tuple['Var', ...] = (), 
                 operation: str = '', label: str = ''):
        self.x = x
        self.grad = 0.0
        self._backward = lambda: None
        self.previous = set(children)
        self.operation = operation
        self.label = label
    
    def __repr__(self) -> str:
        return f"Var({self.x})"
    
    def __add__(self, other: Union['Var', float, int]) -> 'Var':
        other = other if isinstance(other, Var) else Var(float(other))
        out = Var(self.x + other.x, (self, other), "+")
        
        def add_backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = add_backward
        return out
    
    def __radd__(self, other: Union[float, int]) -> 'Var':
        return self + other
    
    def __sub__(self, other: Union['Var', float, int]) -> 'Var':
        return self + (-other)
    
    def __rsub__(self, other: Union[float, int]) -> 'Var':
        return (-self) + other
    
    def __neg__(self) -> 'Var':
        return self * -1
    
    def __mul__(self, other: Union['Var', float, int]) -> 'Var':
        other = other if isinstance(other, Var) else Var(float(other))
        out = Var(self.x * other.x, (self, other), "*")
        
        def mul_backward():
            self.grad += other.x * out.grad
            other.grad += self.x * out.grad
        
        out._backward = mul_backward
        return out
    
    def __rmul__(self, other: Union[float, int]) -> 'Var':
        return self * other
    
    def __truediv__(self, other: Union['Var', float, int]) -> 'Var':
        return self * (other ** -1)
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Var':
        return other * (self ** -1)
    
    def tanh(self) -> 'Var':
        t = (math.exp(2*self.x) - 1) / (math.exp(2*self.x) + 1)
        out = Var(t, (self,), 'tanh')
        
        def tanh_backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = tanh_backward
        return out
    
    def relu(self) -> 'Var':
        out = Var(max(0, self.x), (self,), 'relu')
        
        def relu_backward():
            self.grad += (out.x > 0) * out.grad
        
        out._backward = relu_backward
        return out
    
    def exp(self) -> 'Var':
        x = self.x
        out = Var(math.exp(x), (self,), 'exp')
        
        def exp_backward():
            self.grad += math.exp(x) * out.grad
        
        out._backward = exp_backward
        return out
    
    def log(self) -> 'Var':
        x = self.x
        out = Var(math.log(x), (self,), 'log')
        
        def log_backward():
            self.grad += (1 / x) * out.grad
        
        out._backward = log_backward
        return out
    
    def __pow__(self, power: Union[int, float]) -> 'Var':
        assert isinstance(power, (int, float)), "Power must be a number"
        out = Var(self.x ** power, (self,), f'**{power}')
        
        def pow_backward():
            self.grad += power * (self.x ** (power - 1)) * out.grad
        
        out._backward = pow_backward
        return out
    
    def backward(self) -> None:
        """
        Compute gradients for all variables in the computational graph.
        
        This method performs a topological sort of the computational graph
        and then computes gradients using the chain rule in reverse order.
        """
        ordered = []
        visited = set()
        
        def order_topo(v: 'Var') -> None:
            if v not in visited:
                visited.add(v)
                for child in v.previous:
                    order_topo(child)
                ordered.append(v)
        
        order_topo(self)
        
        self.grad = 1.0
        
        for node in reversed(ordered):
            node._backward()


def draw_computation_graph(root: Var, forward: bool = False):
    """
    Generate a visualization of the computational graph.
    
    This function creates a Graphviz diagram showing the computational
    graph structure, which is useful for debugging and understanding
    the flow of computations.
    
    Args:
        root: The root variable of the computational graph
        forward: If True, show forward pass values; if False, show gradients
    
    Returns:
        A Graphviz Digraph object that can be rendered or saved
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("graphviz package is required for visualization. "
                         "Install with: pip install graphviz")
    
    dot = Digraph(format='svg', graph_attr={'rankdir': 'TBT'})
    
    nodes, edges = set(), set()
    
    def build(v: Var) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v.previous:
                edges.add((child, v))
                build(child)
    
    build(root)
    
    if forward:
        for n in nodes:
            nid = str(id(n))
            dot.node(name=nid, 
                    label=f"{{ {n.label} | {n.x:.4f} | grad {n.grad:.4f}}}", 
                    shape='record')
            if n.operation:
                dot.node(name=nid + n.operation, label=n.operation)
                dot.edge(nid + n.operation, nid)
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2.operation)
    else:
        for n in nodes:
            nid = str(id(n))
            dot.node(name=nid, 
                    label=f"{{∂ {n.label} / ∂ {root.label} | grad {n.grad:.4f}}}", 
                    shape='record')
            if n.operation:
                dot.node(name=nid + n.operation, label=n.operation)
                dot.edge(nid, nid + n.operation)
        
        for n1, n2 in edges:
            dot.edge(str(id(n2)) + n2.operation, str(id(n1)))
    
    return dot
