"""
Optimization Algorithms

This module provides various optimization algorithms for training neural networks,
including Stochastic Gradient Descent (SGD) with learning rate scheduling.
"""

from typing import List, Optional, Callable
from .autograd import Var
import math


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    This optimizer updates model parameters using the computed gradients
    and a learning rate. It supports learning rate scheduling for
    better convergence.
    """
    
    def __init__(self, params: List[Var], lr: float = 0.01, 
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self.learning_rate = lr
        self.params = params
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        if momentum > 0:
            self.velocity = [0.0] * len(params)
    
    def step(self, schedule: Optional[Callable] = None) -> None:
        if schedule:
            self.learning_rate *= schedule
        
        for i, p in enumerate(self.params):
            if self.momentum > 0:
                self.velocity[i] = (self.momentum * self.velocity[i] + 
                                  self.learning_rate * p.grad)
                if self.weight_decay > 0:
                    self.velocity[i] += self.weight_decay * p.x
                
                p.x += -self.velocity[i]
            else:
                if self.weight_decay > 0:
                    p.x += -self.learning_rate * (p.grad + self.weight_decay * p.x)
                else:
                    p.x += -self.learning_rate * p.grad
    
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


class Adam:
    """
    Adam optimizer with adaptive learning rates.
    
    Adam combines the benefits of AdaGrad and RMSprop, providing
    adaptive learning rates for each parameter based on estimates
    of first and second moments of the gradients.
    """
    
    def __init__(self, params: List[Var], lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.learning_rate = lr
        self.params = params
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0
    
    def step(self) -> None:
        self.t += 1
        
        for i, p in enumerate(self.params):
            if self.weight_decay > 0:
                p.grad += self.weight_decay * p.x
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.x += -self.learning_rate * m_hat / (v_hat ** 0.5 + self.eps)
    
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


class RMSprop:
    """
    RMSprop optimizer with adaptive learning rates.
    
    RMSprop adapts the learning rate by dividing it by an exponentially
    decaying average of squared gradients, which helps with training
    stability in deep networks.
    """
    
    def __init__(self, params: List[Var], lr: float = 0.01, 
                 alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.learning_rate = lr
        self.params = params
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.v = [0.0] * len(params)
    
    def step(self) -> None:
        for i, p in enumerate(self.params):
            if self.weight_decay > 0:
                p.grad += self.weight_decay * p.x
            
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (p.grad ** 2)
            
            p.x += -self.learning_rate * p.grad / (self.v[i] ** 0.5 + self.eps)
    
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


class LearningRateScheduler:
    """
    Learning rate scheduler for adaptive learning rate adjustment.
    
    This class provides various scheduling strategies to adjust
    the learning rate during training for better convergence.
    """
    
    def __init__(self, optimizer, strategy: str = 'step', **kwargs):
        self.optimizer = optimizer
        self.strategy = strategy
        self.kwargs = kwargs
        self.epoch = 0
    
    def step(self) -> None:
        if self.strategy == 'step':
            self._step_decay()
        elif self.strategy == 'exponential':
            self._exponential_decay()
        elif self.strategy == 'cosine':
            self._cosine_decay()
        
        self.epoch += 1
    
    def _step_decay(self) -> None:
        step_size = self.kwargs.get('step_size', 30)
        gamma = self.kwargs.get('gamma', 0.1)
        
        if self.epoch > 0 and self.epoch % step_size == 0:
            self.optimizer.learning_rate *= gamma
    
    def _exponential_decay(self) -> None:
        gamma = self.kwargs.get('gamma', 0.95)
        self.optimizer.learning_rate *= gamma
    
    def _cosine_decay(self) -> None:
        max_epochs = self.kwargs.get('max_epochs', 100)
        min_lr = self.kwargs.get('min_lr', 1e-6)
        
        cos_epoch = min(self.epoch, max_epochs)
        cos_decay = 0.5 * (1 + math.cos(math.pi * cos_epoch / max_epochs))
        
        initial_lr = self.kwargs.get('initial_lr', self.optimizer.learning_rate)
        self.optimizer.learning_rate = min_lr + (initial_lr - min_lr) * cos_decay

