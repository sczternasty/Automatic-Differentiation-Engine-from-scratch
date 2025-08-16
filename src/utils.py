"""
Utility Functions and Visualization

This module provides utility functions for data handling, visualization,
and common operations used throughout the autograd engine.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
from .autograd import Var


def visualize_computation_graph(root: Var, forward: bool = False, 
                              save_path: Optional[str] = None) -> None:
    """
    Visualize the computational graph and optionally save it.
    
    This function creates a visual representation of the computational
    graph using Graphviz, which is useful for debugging and understanding
    the flow of computations.
    
    Args:
        root: The root variable of the computational graph
        forward: If True, show forward pass values; if False, show gradients
        save_path: Optional path to save the visualization (without extension)
    """
    try:
        from graphviz import Digraph
        from .autograd import draw_computation_graph
        
        dot = draw_computation_graph(root, forward)
        
        if save_path:
            dot.render(save_path, view=False, format='png')
            print(f"Computation graph saved to {save_path}.png")
        else:
            dot.render("computation_graph", view=False, format='png')
            print("Computation graph saved to computation_graph.png")
            
    except ImportError:
        print("Warning: graphviz package not available. Install with: pip install graphviz")
        print("Cannot visualize computation graph.")


def plot_training_progress(losses: List[float], accuracies: Optional[List[float]] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot training progress including loss and accuracy curves.
    
    Args:
        losses: List of loss values over training epochs
        accuracies: Optional list of accuracy values over training epochs
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2 if accuracies else 1, figsize=(12, 5))
    
    if accuracies:
        axes[0].plot(losses, 'b-', label='Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(accuracies, 'r-', label='Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes.plot(losses, 'b-', label='Training Loss')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.set_title('Training Loss')
        axes.legend()
        axes.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to {save_path}")
    
    plt.show()


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, 
                          save_path: Optional[str] = None) -> None:
    """
    Plot decision boundary for 2D classification problems.
    
    Args:
        model: Trained neural network model
        X: Input features (2D array)
        y: Target labels
        save_path: Optional path to save the plot
    """
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    scores = model(Xmesh.tolist())
    Z = np.array([s.x > 0 for s in scores])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary')
    
    plt.colorbar(contour, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decision boundary plot saved to {save_path}")
    
    plt.show()


def evaluate_model(model, X_test: List[List[float]], y_test: List[Union[float, int]],
                  task: str = 'classification') -> dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained neural network model
        X_test: Test input features
        y_test: Test target values
        task: Task type ('classification' or 'regression')
    
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model(X_test)
    y_pred_values = [pred.x for pred in y_pred]
    
    if task == 'classification':
        y_pred_binary = [1 if pred > 0 else 0 for pred in y_pred_values]
        accuracy = sum(1 for pred, true in zip(y_pred_binary, y_test) if pred == true) / len(y_test)
        
        tp = sum(1 for pred, true in zip(y_pred_binary, y_test) if pred == 1 and true == 1)
        fp = sum(1 for pred, true in zip(y_pred_binary, y_test) if pred == 1 and true == 0)
        fn = sum(1 for pred, true in zip(y_pred_binary, y_test) if pred == 0 and true == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    elif task == 'regression':
        mse = np.mean([(pred - true) ** 2 for pred, true in zip(y_pred_values, y_test)])
        mae = np.mean([abs(pred - true) for pred, true in zip(y_pred_values, y_test)])
        rmse = np.sqrt(mse)
        
        y_mean = np.mean(y_test)
        ss_tot = sum((true - y_mean) ** 2 for true in y_test)
        ss_res = sum((true - pred) ** 2 for true, pred in zip(y_test, y_pred_values))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared
        }
    
    else:
        raise ValueError("Task must be 'classification' or 'regression'")


def normalize_data(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
    """
    Normalize input data using various methods.
    
    Args:
        X: Input data array
        method: Normalization method ('standard', 'minmax', 'robust')
    
    Returns:
        Tuple of normalized data and normalization parameters
    """
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    
    elif method == 'robust':
        median = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        X_norm = (X - median) / (iqr + 1e-8)
        params = {'median': median, 'iqr': iqr}
    
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    return X_norm, params


def apply_normalization(X: np.ndarray, params: dict, method: str) -> np.ndarray:
    """
    Apply normalization to new data using pre-computed parameters.
    
    Args:
        X: New data to normalize
        params: Normalization parameters from normalize_data
        method: Normalization method used
    
    Returns:
        Normalized data
    """
    if method == 'standard':
        return (X - params['mean']) / (params['std'] + 1e-8)
    elif method == 'minmax':
        return (X - params['min']) / (params['max'] - params['min'] + 1e-8)
    elif method == 'robust':
        return (X - params['median']) / (params['iqr'] + 1e-8)
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               val_size: float = 0.0, random_state: Optional[int] = None) -> Tuple:
    """
    Split data into training, validation, and test sets.
    
    Args:
        X: Input features
        y: Target values
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    test_split = int(n_samples * test_size)
    val_split = int(n_samples * val_size)
    
    test_indices = indices[:test_split]
    val_indices = indices[test_split:test_split + val_split]
    train_indices = indices[test_split + val_split:]
    
    X_test, y_test = X[test_indices], y[test_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_train, y_train = X[train_indices], y[train_indices]
    
    if val_size == 0:
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test


def print_model_summary(model) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: Neural network model to summarize
    """
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    
    total_params = 0
    trainable_params = 0
    
    print(f"Model: {type(model).__name__}")
    print(f"Architecture: {model}")
    print()
    
    for param in model.parameters():
        total_params += 1
        if hasattr(param, 'grad'):
            trainable_params += 1
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("=" * 50)

