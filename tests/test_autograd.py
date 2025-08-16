"""
Unit tests for the autograd engine.

This module tests the core functionality of the automatic differentiation
engine, including variable operations, gradient computation, and mathematical
functions.
"""

import sys
import os
import pytest
import math

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd import Var


class TestVar:
    """Test cases for the Var class."""
    
    def test_var_initialization(self):
        """Test variable initialization."""
        v = Var(5.0, label='test')
        assert v.x == 5.0
        assert v.grad == 0.0
        assert v.label == 'test'
        assert v.operation == ''
        assert len(v.previous) == 0
    
    def test_var_repr(self):
        """Test string representation."""
        v = Var(3.14)
        assert str(v) == "Var(3.14)"
    
    def test_addition(self):
        """Test addition operations."""
        a = Var(2.0)
        b = Var(3.0)
        c = a + b
        
        assert c.x == 5.0
        assert c.operation == '+'
        assert len(c.previous) == 2
        
        # Test gradient computation
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0
    
    def test_scalar_addition(self):
        """Test addition with scalars."""
        a = Var(2.0)
        c = a + 3.0
        
        assert c.x == 5.0
        c.backward()
        assert a.grad == 1.0
    
    def test_right_addition(self):
        """Test right addition (scalar + variable)."""
        a = Var(2.0)
        c = 3.0 + a
        
        assert c.x == 5.0
        c.backward()
        assert a.grad == 1.0
    
    def test_subtraction(self):
        """Test subtraction operations."""
        a = Var(5.0)
        b = Var(3.0)
        c = a - b
        
        assert c.x == 2.0
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0
    
    def test_multiplication(self):
        """Test multiplication operations."""
        a = Var(2.0)
        b = Var(3.0)
        c = a * b
        
        assert c.x == 6.0
        c.backward()
        assert a.grad == 3.0  # ∂c/∂a = b
        assert b.grad == 2.0  # ∂c/∂b = a
    
    def test_scalar_multiplication(self):
        """Test multiplication with scalars."""
        a = Var(2.0)
        c = a * 3.0
        
        assert c.x == 6.0
        c.backward()
        assert a.grad == 3.0
    
    def test_division(self):
        """Test division operations."""
        a = Var(6.0)
        b = Var(2.0)
        c = a / b
        
        assert c.x == 3.0
        c.backward()
        assert abs(a.grad - 0.5) < 1e-10  # ∂c/∂a = 1/b
        assert abs(b.grad - (-1.5)) < 1e-10  # ∂c/∂b = -a/b²
    
    def test_power(self):
        """Test power operations."""
        a = Var(2.0)
        c = a ** 3
        
        assert c.x == 8.0
        c.backward()
        assert a.grad == 12.0  # ∂c/∂a = 3 * a²
    
    def test_negation(self):
        """Test negation."""
        a = Var(3.0)
        c = -a
        
        assert c.x == -3.0
        c.backward()
        assert a.grad == -1.0
    
    def test_relu(self):
        """Test ReLU activation function."""
        # Positive input
        a = Var(2.0)
        c = a.relu()
        assert c.x == 2.0
        c.backward()
        assert a.grad == 1.0
        
        # Negative input
        a.grad = 0.0
        a.x = -1.0
        c = a.relu()
        assert c.x == 0.0
        c.backward()
        assert a.grad == 0.0
    
    def test_tanh(self):
        """Test tanh activation function."""
        a = Var(0.0)
        c = a.tanh()
        assert c.x == 0.0
        c.backward()
        assert a.grad == 1.0  # ∂tanh(0)/∂x = 1 - tanh²(0) = 1
    
    def test_exp(self):
        """Test exponential function."""
        a = Var(0.0)
        c = a.exp()
        assert c.x == 1.0
        c.backward()
        assert a.grad == 1.0  # ∂exp(0)/∂x = exp(0) = 1
    
    def test_log(self):
        """Test logarithm function."""
        a = Var(1.0)
        c = a.log()
        assert c.x == 0.0
        c.backward()
        assert a.grad == 1.0  # ∂log(1)/∂x = 1/1 = 1
    
    def test_complex_expression(self):
        """Test a complex mathematical expression."""
        x = Var(2.0)
        y = Var(3.0)
        
        # f(x,y) = x² + y²
        f = x**2 + y**2
        
        assert f.x == 13.0  # 2² + 3² = 4 + 9 = 13
        
        f.backward()
        assert x.grad == 4.0  # ∂f/∂x = 2x = 4
        assert y.grad == 6.0  # ∂f/∂y = 2y = 6
    
    def test_chain_rule(self):
        """Test chain rule in gradient computation."""
        x = Var(2.0)
        
        # f(x) = exp(x²)
        f = (x**2).exp()
        
        assert f.x == math.exp(4)  # exp(2²) = exp(4)
        
        f.backward()
        # ∂f/∂x = exp(x²) * 2x = exp(4) * 4
        expected_grad = math.exp(4) * 4
        assert abs(x.grad - expected_grad) < 1e-10
    
    def test_multiple_operations(self):
        """Test multiple operations in sequence."""
        a = Var(1.0)
        b = Var(2.0)
        c = Var(3.0)
        
        # f = (a + b) * c
        f = (a + b) * c
        
        assert f.x == 9.0  # (1 + 2) * 3 = 3 * 3 = 9
        
        f.backward()
        assert a.grad == 3.0  # ∂f/∂a = c = 3
        assert b.grad == 3.0  # ∂f/∂b = c = 3
        assert c.grad == 3.0  # ∂f/∂c = (a + b) = 3
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly."""
        a = Var(2.0)
        b = Var(3.0)
        
        # First computation: f1 = a * b
        f1 = a * b
        f1.backward()
        
        assert a.grad == 3.0
        assert b.grad == 2.0
        
        # Second computation: f2 = a + b
        f2 = a + b
        f2.backward()
        
        # Gradients should accumulate
        assert a.grad == 4.0  # 3.0 + 1.0
        assert b.grad == 3.0  # 2.0 + 1.0
    
    def test_zero_grad(self):
        """Test that gradients can be reset."""
        a = Var(2.0)
        b = Var(3.0)
        
        f = a * b
        f.backward()
        
        assert a.grad != 0.0
        assert b.grad != 0.0
        
        # Reset gradients
        a.grad = 0.0
        b.grad = 0.0
        
        assert a.grad == 0.0
        assert b.grad == 0.0


class TestMathematicalProperties:
    """Test mathematical properties and edge cases."""
    
    def test_commutativity(self):
        """Test that addition and multiplication are commutative."""
        a = Var(2.0)
        b = Var(3.0)
        
        # Addition
        c1 = a + b
        c2 = b + a
        assert c1.x == c2.x
        
        # Multiplication
        d1 = a * b
        d2 = b * a
        assert d1.x == d2.x
    
    def test_associativity(self):
        """Test that addition and multiplication are associative."""
        a = Var(1.0)
        b = Var(2.0)
        c = Var(3.0)
        
        # Addition
        result1 = (a + b) + c
        result2 = a + (b + c)
        assert abs(result1.x - result2.x) < 1e-10
        
        # Multiplication
        result3 = (a * b) * c
        result4 = a * (b * c)
        assert abs(result3.x - result4.x) < 1e-10
    
    def test_distributivity(self):
        """Test distributive property: a * (b + c) = a*b + a*c."""
        a = Var(2.0)
        b = Var(3.0)
        c = Var(4.0)
        
        left = a * (b + c)
        right = a * b + a * c
        
        assert abs(left.x - right.x) < 1e-10
    
    def test_identity_elements(self):
        """Test identity elements for addition and multiplication."""
        a = Var(5.0)
        
        # Addition identity: a + 0 = a
        assert (a + 0).x == a.x
        
        # Multiplication identity: a * 1 = a
        assert (a * 1).x == a.x
    
    def test_inverse_elements(self):
        """Test inverse elements."""
        a = Var(5.0)
        
        # Additive inverse: a + (-a) = 0
        assert (a + (-a)).x == 0.0
        
        # Multiplicative inverse: a * (1/a) = 1
        assert abs((a * (1/a)).x - 1.0) < 1e-10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


