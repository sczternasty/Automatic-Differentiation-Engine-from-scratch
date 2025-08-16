#!/usr/bin/env python3
"""
Basic Automatic Differentiation Example

This example demonstrates the core functionality of the autograd engine,
showing how to create variables, perform mathematical operations, and
compute gradients automatically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd import Var, draw_computation_graph


def basic_operations_example():
    print("=" * 50)
    print("BASIC AUTOGRAD EXAMPLE")
    print("=" * 50)
    
    a = Var(2.0, label='a')
    b = Var(3.0, label='b')
    c = Var(4.0, label='c')
    
    print(f"Initial values: a={a.x}, b={b.x}, c={c.x}")
    
    d = a + b * c
    d.label = 'd'
    
    print(f"d = a + b * c = {a.x} + {b.x} * {c.x} = {d.x}")
    
    d.backward()
    
    print("\nGradients after backward pass:")
    print(f"∂d/∂a = {a.grad}")
    print(f"∂d/∂b = {b.grad}")
    print(f"∂d/∂c = {c.grad}")
    
    print("\nManual verification:")
    print(f"∂d/∂a = ∂(a + b*c)/∂a = 1 = {a.grad}")
    print(f"∂d/∂b = ∂(a + b*c)/∂b = c = {c.x} = {b.grad}")
    print(f"∂d/∂c = ∂(a + b*c)/∂c = b = {b.x} = {c.grad}")
    
    return d


def activation_functions_example():
    print("\n" + "=" * 50)
    print("ACTIVATION FUNCTIONS EXAMPLE")
    print("=" * 50)
    
    x = Var(1.0, label='x')
    
    relu_out = x.relu()
    tanh_out = x.tanh()
    exp_out = x.exp()
    log_out = (x + 1).log()
    
    print(f"Input: x = {x.x}")
    print(f"ReLU(x) = {relu_out.x}")
    print(f"tanh(x) = {tanh_out.x}")
    print(f"exp(x) = {exp_out.x}")
    print(f"log(x+1) = {log_out.x}")
    
    relu_out.backward()
    print(f"\n∂ReLU(x)/∂x = {x.grad}")
    
    x.grad = 0.0
    tanh_out.backward()
    print(f"∂tanh(x)/∂x = {x.grad}")
    
    x.grad = 0.0
    exp_out.backward()
    print(f"∂exp(x)/∂x = {exp_out.x} = {x.grad}")
    
    x.grad = 0.0
    log_out.backward()
    print(f"∂log(x+1)/∂x = 1/(x+1) = {1/(x.x + 1)} = {x.grad}")


def complex_expression_example():
    print("\n" + "=" * 50)
    print("COMPLEX EXPRESSION EXAMPLE")
    print("=" * 50)
    
    x = Var(2.0, label='x')
    y = Var(3.0, label='y')
    
    f = (x**2 + y**3) * (x + y).tanh()
    f.label = 'f'
    
    print(f"f(x,y) = (x² + y³) * tanh(x + y)")
    print(f"f({x.x}, {y.x}) = ({x.x}² + {y.x}³) * tanh({x.x} + {y.x})")
    print(f"f({x.x}, {y.x}) = ({x.x**2} + {y.x**3}) * tanh({x.x + y.x})")
    print(f"f({x.x}, {y.x}) = {x.x**2 + y.x**3} * {(x.x + y.x).__tanh__()}")
    print(f"f({x.x}, {y.x}) = {f.x}")
    
    f.backward()
    
    print(f"\nGradients:")
    print(f"∂f/∂x = {x.grad}")
    print(f"∂f/∂y = {y.grad}")
    
    return f


def visualization_example():
    print("\n" + "=" * 50)
    print("VISUALIZATION EXAMPLE")
    print("=" * 50)
    
    a = Var(1.0, label='a')
    b = Var(2.0, label='b')
    c = a * b + a**2
    c.label = 'c'
    
    print(f"Expression: c = a * b + a² = {a.x} * {b.x} + {a.x}² = {c.x}")
    
    try:
        print("\nGenerating computational graph visualization...")
        dot = draw_computation_graph(c, forward=True)
        print("Computational graph generated successfully!")
        print("You can save it using: dot.render('graph_name', view=True)")
        
        dot.render("examples/computation_graph_example", view=False, format='png')
        print("Graph saved as 'examples/computation_graph_example.png'")
        
    except ImportError:
        print("Graphviz not available. Install with: pip install graphviz")
    except Exception as e:
        print(f"Error generating graph: {e}")


def main():
    print("AUTOMATIC DIFFERENTIATION ENGINE - BASIC EXAMPLES")
    print("=" * 60)
    
    basic_operations_example()
    activation_functions_example()
    complex_expression_example()
    visualization_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


