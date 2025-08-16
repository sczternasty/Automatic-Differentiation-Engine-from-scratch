#!/usr/bin/env python3
"""
PyTorch Cross-Check Runner

This script runs the PyTorch cross-checks to validate the correctness
of the autograd engine implementation.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.pytorch_check import run_all_checks
    print("Starting PyTorch cross-checks...")
    success = run_all_checks()
    
    if success:
        print("\nAll PyTorch cross-checks passed!")
        sys.exit(0)
    else:
        print("\nSome PyTorch cross-checks failed!")
        sys.exit(1)
        
except ImportError as e:
    print(f"Error importing PyTorch check module: {e}")
    print("Make sure PyTorch is installed: pip install torch")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
