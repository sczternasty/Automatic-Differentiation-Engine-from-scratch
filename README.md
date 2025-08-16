# Automatic Differentiation Engine

A complete implementation of automatic differentiation (autograd) and neural network training from first principles, built entirely in Python without external deep learning frameworks.

## Overview

This project implements a complete automatic differentiation engine that enables training neural networks using backpropagation. It demonstrates:

- **Automatic Differentiation**: Forward and backward pass implementations
- **Neural Network Architecture**: Multi-layer perceptrons with various activation functions
- **Optimization Algorithms**: Stochastic Gradient Descent with learning rate scheduling
- **Loss Functions**: MSE, Binary Cross-Entropy, and Max-Margin losses
- **Regularization**: L2 regularization techniques
- **Computational Graph Visualization**: Dynamic graph generation for debugging
- **PyTorch Cross-Checking**: Validation against PyTorch for correctness verification

## Features

### Core Autograd Engine
- **Variable Class**: Implements automatic differentiation with gradient computation
- **Mathematical Operations**: Addition, multiplication, division, power, trigonometric functions
- **Activation Functions**: ReLU, Tanh, Exponential, Logarithm
- **Topological Sorting**: Efficient backward pass through computational graphs

### Neural Network Components
- **Linear Layers**: Fully connected layers with configurable input/output dimensions
- **Activation Functions**: ReLU and Tanh as separate modules
- **Multi-Layer Perceptron**: Configurable architecture with sequential layers
- **Parameter Management**: Automatic gradient zeroing and parameter collection

### Training Infrastructure
- **Loss Functions**: Multiple loss implementations for different problem types
- **Optimizers**: SGD, Adam, and RMSprop with learning rate scheduling
- **Regularization**: L2 regularization to prevent overfitting
- **Data Handling**: Support for various dataset formats

### Validation and Testing
- **PyTorch Cross-Checks**: Comprehensive validation against PyTorch implementations
- **Gradient Verification**: Ensures mathematical correctness of all operations
- **Performance Benchmarking**: Comparison with industry-standard frameworks

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── autograd.py          # Core automatic differentiation engine
│   ├── nn.py               # Neural network modules
│   ├── losses.py           # Loss function implementations
│   ├── optimizers.py       # Optimization algorithms
│   ├── utils.py            # Utility functions and visualization
│   └── pytorch_check.py    # PyTorch cross-checking and validation
├── examples/
│   ├── basic_autograd.py   # Basic autograd examples
│   └── autograd_engine.ipynb  # Jupyter notebook examples
├── tests/
│   └── test_autograd.py    # Unit tests for autograd engine
├── requirements.txt         # Python dependencies
├── demo.py                 # Demonstration script
└── check_pytorch.py        # PyTorch validation runner
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Automatic-Differentiation-Engine-from-scratch

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python run_tests.py
```

## Usage Examples

### Basic Automatic Differentiation
```python
from src.autograd import Var

# Create variables
a = Var(2.0, label='a')
b = Var(3.0, label='b')
c = a * b + 2

# Compute gradients
c.backward()
print(f"∂c/∂a = {a.grad}")  # Output: ∂c/∂a = 3.0
print(f"∂c/∂b = {b.grad}")  # Output: ∂c/∂b = 2.0
```

### Training a Neural Network
```python
from src.nn import MLP
from src.losses import MSELoss
from src.optimizers import SGD

# Create model and optimizer
model = MLP()
optimizer = SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
    y_pred = model(X)
    loss = MSELoss(y_pred, y_true)
    
    model.zero_grad()
    loss.backward()
    optimizer.step()
```

### PyTorch Cross-Checking
```python
from src.pytorch_check import run_all_checks

# Validate implementation against PyTorch
success = run_all_checks()
if success:
    print("All checks passed! Implementation is correct.")
```

## Testing

```bash
# Run all tests
python run_tests.py

# Run PyTorch cross-checks
python check_pytorch.py

# Run demo
python demo.py
```

## Performance

- **Training Speed**: Comparable to basic PyTorch implementations for small networks
- **Memory Efficiency**: Optimized for clarity with clean code structure
- **Scalability**: Designed for understanding, can be extended for production use
- **Correctness**: Validated against PyTorch for mathematical accuracy

## Technical Implementation

### Automatic Differentiation
- **Forward Mode**: Computes values and builds computational graph
- **Backward Mode**: Efficiently computes gradients using chain rule
- **Memory Management**: Automatic cleanup of intermediate computations

### Neural Network Architecture
- **Modular Design**: Each component is a separate, testable module
- **Extensible**: Easy to add new activation functions and layer types
- **Efficient**: Optimized forward and backward passes

### PyTorch Validation
- **Gradient Verification**: Ensures mathematical correctness across all operations
- **Performance Comparison**: Benchmarks against industry-standard implementation
- **Regression Testing**: Catches implementation errors and performance regressions

