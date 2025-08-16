
from .autograd import Var, draw_computation_graph
from .nn import Module, Perceptron, Linear, ReLU, Tanh, MLP, Moon_MLP, CH_MLP
from .losses import (
    MSELoss, BCELoss, MaxMarginLoss, L2Regularization,
    CrossEntropyLoss, HuberLoss, FocalLoss
)
from .optimizers import SGD, Adam, RMSprop, LearningRateScheduler
from .utils import (
    visualize_computation_graph, plot_training_progress, plot_decision_boundary,
    evaluate_model, normalize_data, apply_normalization, split_data, print_model_summary
)
from .pytorch_check import (
    check_basic_operations, check_activation_functions, 
    check_neural_network, check_optimization, run_all_checks
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'Var', 'draw_computation_graph',
    'Module', 'Perceptron', 'Linear', 'ReLU', 'Tanh', 'MLP', 'Moon_MLP', 'CH_MLP',
    'MSELoss', 'BCELoss', 'MaxMarginLoss', 'L2Regularization',
    'CrossEntropyLoss', 'HuberLoss', 'FocalLoss',
    'SGD', 'Adam', 'RMSprop', 'LearningRateScheduler',
    'visualize_computation_graph', 'plot_training_progress', 'plot_decision_boundary',
    'evaluate_model', 'normalize_data', 'apply_normalization', 'split_data', 'print_model_summary',
    'check_basic_operations', 'check_activation_functions', 
    'check_neural_network', 'check_optimization', 'run_all_checks'
]

