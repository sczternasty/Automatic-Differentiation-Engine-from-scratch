"""
PyTorch Cross-Check Module

This module provides functionality to validate the correctness of the autograd engine
by comparing results with PyTorch implementations on various tasks.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from .autograd import Var

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


def check_basic_operations():
    """Check basic mathematical operations against PyTorch."""
    if not PYTORCH_AVAILABLE:
        return False, "PyTorch not available"
    
    print("Checking basic operations...")
    
    a_val, b_val, c_val = 2.0, 3.0, 4.0
    
    a = Var(a_val, label='a')
    b = Var(b_val, label='b')
    c = Var(c_val, label='c')
    
    d = a + b * c
    d.backward()
    
    a_torch = torch.tensor(a_val, requires_grad=True)
    b_torch = torch.tensor(b_val, requires_grad=True)
    c_torch = torch.tensor(c_val, requires_grad=True)
    
    d_torch = a_torch + b_torch * c_torch
    d_torch.backward()
    
    value_match = abs(d.x - d_torch.item()) < 1e-6
    grad_a_match = abs(a.grad - a_torch.grad.item()) < 1e-6
    grad_b_match = abs(b.grad - b_torch.grad.item()) < 1e-6
    grad_c_match = abs(c.grad - c_torch.grad.item()) < 1e-6
    
    print(f"Value: {d.x:.6f} vs PyTorch {d_torch.item():.6f} - {'PASS' if value_match else 'FAIL'}")
    print(f"∂d/∂a: {a.grad:.6f} vs PyTorch {a_torch.grad.item():.6f} - {'PASS' if grad_a_match else 'FAIL'}")
    print(f"∂d/∂b: {b.grad:.6f} vs PyTorch {b_torch.grad.item():.6f} - {'PASS' if grad_b_match else 'FAIL'}")
    print(f"∂d/∂c: {c.grad:.6f} vs PyTorch {c_torch.grad.item():.6f} - {'PASS' if grad_c_match else 'FAIL'}")
    
    success = value_match and grad_a_match and grad_b_match and grad_c_match
    return success, "Basic operations check completed"


def check_activation_functions():
    """Check activation functions against PyTorch."""
    if not PYTORCH_AVAILABLE:
        return False, "PyTorch not available"
    
    print("Checking activation functions...")
    
    x_val = 1.5
    
    x = Var(x_val, label='x')
    relu_out = x.relu()
    tanh_out = x.tanh()
    exp_out = x.exp()
    
    relu_out.backward()
    relu_grad = x.grad
    
    x.grad = 0.0
    tanh_out.backward()
    tanh_grad = x.grad
    
    x.grad = 0.0
    exp_out.backward()
    exp_grad = x.grad
    
    x_torch = torch.tensor(x_val, requires_grad=True)
    
    relu_torch = torch.relu(x_torch)
    relu_torch.backward()
    relu_grad_torch = x_torch.grad.item()
    
    x_torch.grad.zero_()
    tanh_torch = torch.tanh(x_torch)
    tanh_torch.backward()
    tanh_grad_torch = x_torch.grad.item()
    
    x_torch.grad.zero_()
    exp_torch = torch.exp(x_torch)
    exp_torch.backward()
    exp_grad_torch = x_torch.grad.item()
    
    relu_match = abs(relu_out.x - relu_torch.item()) < 1e-6
    tanh_match = abs(tanh_out.x - tanh_torch.item()) < 1e-6
    exp_match = abs(exp_out.x - exp_torch.item()) < 1e-6
    
    relu_grad_match = abs(relu_grad - relu_grad_torch) < 1e-6
    tanh_grad_match = abs(tanh_grad - tanh_grad_torch) < 1e-6
    exp_grad_match = abs(exp_grad - exp_grad_torch) < 1e-6
    
    print(f"ReLU: {relu_out.x:.6f} vs PyTorch {relu_torch.item():.6f} - {'PASS' if relu_match else 'FAIL'}")
    print(f"tanh: {tanh_out.x:.6f} vs PyTorch {tanh_torch.item():.6f} - {'PASS' if tanh_match else 'FAIL'}")
    print(f"exp: {exp_out.x:.6f} vs PyTorch {exp_torch.item():.6f} - {'PASS' if exp_match else 'FAIL'}")
    print(f"ReLU grad: {relu_grad:.6f} vs PyTorch {relu_grad_torch:.6f} - {'PASS' if relu_grad_match else 'FAIL'}")
    print(f"tanh grad: {tanh_grad:.6f} vs PyTorch {tanh_grad_torch:.6f} - {'PASS' if tanh_grad_match else 'FAIL'}")
    print(f"exp grad: {exp_grad:.6f} vs PyTorch {exp_grad_torch:.6f} - {'PASS' if exp_grad_match else 'FAIL'}")
    
    success = (relu_match and tanh_match and exp_match and 
               relu_grad_match and tanh_grad_match and exp_grad_match)
    return success, "Activation functions check completed"


def check_neural_network():
    """Check neural network forward and backward pass against PyTorch."""
    if not PYTORCH_AVAILABLE:
        return False, "PyTorch not available"
    
    print("Checking neural network...")
    
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    y = [1.0, 0.0]
    
    from .nn import MLP
    from .losses import MSELoss
    from .optimizers import SGD
    
    model = MLP([3, 4, 4, 1])
    optimizer = SGD(model.parameters(), lr=0.01)
    
    y_pred = model(X)
    loss = MSELoss(y_pred, y)
    
    model.zero_grad()
    loss.backward()
    
    torch_model = nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 4),
        nn.Tanh(),
        nn.Linear(4, 1)
    )
    
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            if i < len(list(torch_model.parameters())):
                torch_param = list(torch_model.parameters())[i]
                torch_param.copy_(torch.tensor(param.x))
    
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
    
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    y_pred_torch = torch_model(X_torch)
    loss_torch = nn.MSELoss()(y_pred_torch, y_torch)
    
    torch_optimizer.zero_grad()
    loss_torch.backward()
    
    loss_match = abs(loss.x - loss_torch.item()) < 1e-4
    print(f"Loss: {loss.x:.6f} vs PyTorch {loss_torch.item():.6f} - {'PASS' if loss_match else 'FAIL'}")
    
    your_params = model.parameters()
    torch_params = list(torch_model.parameters())
    
    grad_matches = []
    for i in range(min(3, len(your_params), len(torch_params))):
        your_grad = your_params[i].grad
        torch_grad = torch_params[i].grad.mean().item() if torch_params[i].grad is not None else 0.0
        match = abs(your_grad - torch_grad) < 1e-4
        grad_matches.append(match)
        print(f"Param {i} grad: {your_grad:.6f} vs PyTorch {torch_grad:.6f} - {'PASS' if match else 'FAIL'}")
    
    success = loss_match and all(grad_matches)
    return success, "Neural network check completed"


def check_optimization():
    """Check optimization algorithms against PyTorch."""
    if not PYTORCH_AVAILABLE:
        return False, "PyTorch not available"
    
    print("Checking optimization...")
    
    x = Var(2.0, label='x')
    f = x**2 + 2*x + 1
    
    from .optimizers import SGD
    optimizer = SGD([x], lr=0.1)
    
    losses = []
    for step in range(20):
        f = x**2 + 2*x + 1
        losses.append(f.x)
        
        optimizer.zero_grad()
        f.backward()
        optimizer.step()
    
    x_torch = torch.tensor(2.0, requires_grad=True)
    torch_optimizer = torch.optim.SGD([x_torch], lr=0.1)
    
    torch_losses = []
    for step in range(20):
        f_torch = x_torch**2 + 2*x_torch + 1
        torch_losses.append(f_torch.item())
        
        torch_optimizer.zero_grad()
        f_torch.backward()
        torch_optimizer.step()
    
    final_x_match = abs(x.x - x_torch.item()) < 1e-3
    final_loss_match = abs(losses[-1] - torch_losses[-1]) < 1e-3
    
    print(f"Final x: {x.x:.6f} vs PyTorch {x_torch.item():.6f} - {'PASS' if final_x_match else 'FAIL'}")
    print(f"Final loss: {losses[-1]:.6f} vs PyTorch {torch_losses[-1]:.6f} - {'PASS' if final_loss_match else 'FAIL'}")
    
    success = final_x_match and final_loss_match
    return success, "Optimization check completed"


def run_all_checks():
    """Run all PyTorch cross-checks."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        return
    
    print("=" * 60)
    print("PYTORCH CROSS-CHECK VALIDATION")
    print("=" * 60)
    
    checks = [
        check_basic_operations,
        check_activation_functions,
        check_neural_network,
        check_optimization
    ]
    
    results = []
    for check in checks:
        try:
            success, message = check()
            results.append((check.__name__, success, message))
        except Exception as e:
            results.append((check.__name__, False, f"Error: {str(e)}"))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, success, message in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
        if not success:
            print(f"  {message}")
        else:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("All checks passed! Your autograd engine matches PyTorch results.")
    else:
        print("Some checks failed. Review the implementation.")
    
    return passed == len(results)


if __name__ == "__main__":
    run_all_checks()
