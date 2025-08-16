"""
Loss Functions

This module provides various loss functions for training neural networks,
including regression losses, classification losses, and regularization terms.
"""

from typing import List, Union
from .autograd import Var


def MSELoss(ypred: List[Var], ytrue: List[Union[float, int]]) -> Var:
    """
    Mean Squared Error loss function.
    
    This loss is commonly used for regression problems and measures
    the average squared difference between predicted and true values.
    
    Args:
        ypred: List of predicted values (Var objects)
        ytrue: List of true target values (scalars)
    
    Returns:
        MSE loss as a Var object
    """
    ytrue_vars = [Var(float(yt)) for yt in ytrue]
    
    squared_errors = [(yp - yt) ** 2 for yp, yt in zip(ypred, ytrue_vars)]
    total_loss = sum(squared_errors) / len(ypred)
    
    return total_loss


def BCELoss(ypred: List[Var], ytrue: List[Union[float, int]]) -> Var:
    """
    Binary Cross-Entropy loss function.
    
    This loss is used for binary classification problems and measures
    the negative log-likelihood of the correct class predictions.
    
    Args:
        ypred: List of predicted probabilities (Var objects)
        ytrue: List of true binary labels (0 or 1)
    
    Returns:
        BCE loss as a Var object
    """
    total_loss = 0
    n = len(ypred)
    
    for yp, yt in zip(ypred, ytrue):
        yp_clamped = Var(min(max(yp.x, 1e-7), 1 - 1e-7))
        
        if yt == 1:
            loss_term = yp_clamped.log()
        else:
            loss_term = (Var(1) - yp_clamped).log()
        
        total_loss += loss_term
    
    return -total_loss / n


def MaxMarginLoss(ypred: List[Var], ytrue: List[Union[float, int]]) -> Var:
    """
    Max-Margin Loss (SVM-style) for classification.
    
    This loss encourages correct predictions to have higher scores
    than incorrect ones by a margin of 1. It's useful for
    binary classification when you want to maximize the margin
    between classes.
    
    Args:
        ypred: List of predicted scores (Var objects)
        ytrue: List of true binary labels (-1 or 1)
    
    Returns:
        Max-margin loss as a Var object
    """
    ytrue_vars = [Var(float(yt)) for yt in ytrue]
    
    losses = [(Var(1) + (-yt * yp)).relu() for yt, yp in zip(ytrue_vars, ypred)]
    
    return sum(losses) / len(ypred)


def L2Regularization(params: List[Var], alpha: float = 1e-4) -> Var:
    """
    L2 Regularization term.
    
    This function adds L2 regularization to prevent overfitting by
    penalizing large parameter values. The regularization term is
    added to the main loss during training.
    
    Args:
        params: List of model parameters (Var objects)
        alpha: Regularization strength (default: 1e-4)
    
    Returns:
        L2 regularization term as a Var object
    """
    regularization_term = alpha * sum((p * p for p in params))
    return regularization_term


def CrossEntropyLoss(ypred: List[Var], ytrue: List[int]) -> Var:
    """
    Cross-Entropy Loss for multi-class classification.
    
    This loss is used for multi-class classification problems and
    measures the negative log-likelihood of the correct class.
    
    Args:
        ypred: List of predicted logits (Var objects)
        ytrue: List of true class indices (integers)
    
    Returns:
        Cross-entropy loss as a Var object
    """
    total_loss = 0
    n = len(ypred)
    
    for yp, yt in zip(ypred, ytrue):
        exp_yp = yp.exp()
        if yt == 1:
            loss_term = exp_yp.log()
        else:
            loss_term = (Var(1) - exp_yp).log()
        
        total_loss += loss_term
    
    return -total_loss / n


def HuberLoss(ypred: List[Var], ytrue: List[Union[float, int]], delta: float = 1.0) -> Var:
    """
    Huber Loss for robust regression.
    
    This loss combines the best properties of MSE and MAE losses.
    It's less sensitive to outliers than MSE and provides smooth
    gradients for optimization.
    
    Args:
        ypred: List of predicted values (Var objects)
        ytrue: List of true target values (scalars)
        delta: Threshold parameter (default: 1.0)
    
    Returns:
        Huber loss as a Var object
    """
    ytrue_vars = [Var(float(yt)) for yt in ytrue]
    
    total_loss = 0
    n = len(ypred)
    
    for yp, yt in zip(ypred, ytrue_vars):
        error = abs(yp - yt)
        
        if error.x <= delta:
            loss_term = Var(0.5) * error ** 2
        else:
            loss_term = delta * error - Var(0.5 * delta ** 2)
        
        total_loss += loss_term
    
    return total_loss / n


def FocalLoss(ypred: List[Var], ytrue: List[Union[float, int]], 
              alpha: float = 1.0, gamma: float = 2.0) -> Var:
    """
    Focal Loss for addressing class imbalance.
    
    This loss down-weights easy examples and focuses training on
    hard examples, which is useful for imbalanced datasets.
    
    Args:
        ypred: List of predicted probabilities (Var objects)
        ytrue: List of true binary labels (0 or 1)
        alpha: Weighting factor for class imbalance (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
    
    Returns:
        Focal loss as a Var object
    """
    total_loss = 0
    n = len(ypred)
    
    for yp, yt in zip(ypred, ytrue):
        yp_clamped = Var(min(max(yp.x, 1e-7), 1 - 1e-7))
        
        if yt == 1:
            pt = yp_clamped
            alpha_t = alpha
        else:
            pt = Var(1) - yp_clamped
            alpha_t = Var(1) - alpha
        
        focal_weight = (Var(1) - pt) ** gamma
        loss_term = -alpha_t * focal_weight * pt.log()
        
        total_loss += loss_term
    
    return total_loss / n

