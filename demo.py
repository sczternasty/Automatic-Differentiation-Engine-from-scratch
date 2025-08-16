#!/usr/bin/env python3
"""
Demo script for the Automatic Differentiation Engine from Scratch

This script demonstrates the key features of the autograd engine,
including basic operations, neural network training, and visualization.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.autograd import Var, draw_computation_graph
from src.nn import MLP, Moon_MLP
from src.losses import MSELoss, MaxMarginLoss, L2Regularization
from src.optimizers import SGD
from src.utils import plot_training_progress, print_model_summary


def demo_basic_autograd():
    print("=" * 60)
    print("DEMO: Basic Automatic Differentiation")
    print("=" * 60)
    
    a = Var(2.0, label='a')
    b = Var(3.0, label='b')
    c = Var(4.0, label='c')

    d = a + b * c
    d.label = 'd'
    
    print(f"Expression: d = a + b * c = {a.x} + {b.x} * {c.x} = {d.x}")

    d.backward()
    
    print("\nGradients:")
    print(f"∂d/∂a = {a.grad}")
    print(f"∂d/∂b = {c.x} = {b.grad}")
    print(f"∂d/∂c = {b.x} = {c.grad}")
    
    return d


def demo_neural_network():
    """Demonstrate neural network training."""
    print("\n" + "=" * 60)
    print("DEMO: Neural Network Training")
    print("=" * 60)
    
    X = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    y = [1.0, -1.0, -1.0, 1.0]
    

    model = MLP()
    print_model_summary(model)

    optimizer = SGD(model.parameters(), lr=0.1)

    losses = []
    print("\nTraining...")
    
    for epoch in range(20):
        y_pred = model(X)
        loss = MSELoss(y_pred, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.x)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.x:.4f}")
    
    print(f"Final loss: {losses[-1]:.4f}")
    
    try:
        plot_training_progress(losses, save_path="training_progress")
    except Exception as e:
        print(f"Could not plot training progress: {e}")
    
    return model, losses


def demo_computation_graph():
    """Demonstrate computational graph visualization."""
    print("\n" + "=" * 60)
    print("DEMO: Computational Graph Visualization")
    print("=" * 60)
    
    x = Var(1.0, label='x')
    y = Var(2.0, label='y')
    z = x * y + x**2
    z.label = 'z'
    
    print(f"Expression: z = x * y + x² = {x.x} * {y.x} + {x.x}² = {z.x}")
    
    try:
        dot = draw_computation_graph(z, forward=True)
        dot.render("demo_computation_graph", view=False, format='png')
        print("Computational graph saved as 'demo_computation_graph.png'")
        print("The graph shows the forward pass with values and operations.")
    except Exception as e:
        print(f"Could not generate graph: {e}")
        print("Make sure graphviz is installed: pip install graphviz")
    
    return z


def demo_moons_classification():
    """Demonstrate classification on the moons dataset."""
    print("\n" + "=" * 60)
    print("DEMO: Moons Classification")
    print("=" * 60)
    
    try:
        from sklearn.datasets import make_moons
        
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
        y = y * 2 - 1
        
        print(f"Dataset shape: {X.shape}")
        print(f"Labels: {np.unique(y)}")
        
        model = Moon_MLP()
        print_model_summary(model)
        
        optimizer = SGD(model.parameters(), lr=1.0)
        
        losses = []
        accuracies = []
        
        print("\nTraining...")
        for epoch in range(50):
            preds = model(X.tolist())
            loss = MaxMarginLoss(preds, y) + L2Regularization(model.parameters())
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            accuracy = [(yi > 0) == (scorei.x > 0) for yi, scorei in zip(y, preds)]
            acc = sum(accuracy) / len(accuracy)
            
            losses.append(loss.x)
            accuracies.append(acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.x:.4f}, Accuracy = {acc*100:.1f}%")
        
        print(f"Final accuracy: {accuracies[-1]*100:.1f}%")
        
        try:
            plot_training_progress(losses, accuracies, save_path="moons_training")
        except Exception as e:
            print(f"Could not plot training progress: {e}")
        
        return model, losses, accuracies
        
    except ImportError:
        print("scikit-learn not available. Install with: pip install scikit-learn")
        return None, None, None


def demo_pytorch_check():
    """Demonstrate PyTorch cross-checking."""
    print("\n" + "=" * 60)
    print("DEMO: PyTorch Cross-Check")
    print("=" * 60)
    
    try:
        from src.pytorch_check import run_all_checks
        
        print("Running PyTorch cross-checks to validate implementation...")
        success = run_all_checks()
        
        if success:
            print("\nAll PyTorch checks passed! Your implementation is correct.")
        else:
            print("\nSome PyTorch checks failed. Review the implementation.")
        
        return success
        
    except ImportError:
        print("PyTorch not available. Install with: pip install torch")
        return False
    except Exception as e:
        print(f"Error running PyTorch checks: {e}")
        return False


def main():
    """Run all demos."""
    print("AUTOMATIC DIFFERENTIATION ENGINE - DEMO")
    
    demo_basic_autograd()
    demo_neural_network()
    demo_computation_graph()
    demo_moons_classification()
    demo_pytorch_check()

    print("All demos completed successfully!")


if __name__ == "__main__":
    main()


