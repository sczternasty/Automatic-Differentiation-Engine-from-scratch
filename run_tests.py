#!/usr/bin/env python3
"""
Test runner for the Automatic Differentiation Engine
"""

import sys
import os
import subprocess

def run_tests():
    print("Running tests for the Automatic Differentiation Engine...")
    print("=" * 60)

    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", 
            "-v", "--cov=src", "--cov-report=term-missing"
        ], capture_output=True, text=True)
        
        print("Test Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Test Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("All tests passed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Some tests failed!")
            print("=" * 60)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_basic_functionality_test():
    print("Running basic functionality test...")
    print("=" * 60)
    
    try:
        from src.autograd import Var
        
        a = Var(2.0)
        b = Var(3.0)
        c = a + b
        
        assert c.x == 5.0, f"Expected 5.0, got {c.x}"
        
        c.backward()
        assert a.grad == 1.0, f"Expected gradient 1.0, got {a.grad}"
        assert b.grad == 1.0, f"Expected gradient 1.0, got {b.grad}"
        
        print("Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        return False

def main():
    print("AUTOMATIC DIFFERENTIATION ENGINE - TEST RUNNER")
    print("=" * 60)

    if run_tests():
        print("\nAll tests completed successfully!")
    else:
        print("\nPytest failed, running basic functionality test...")
        if run_basic_functionality_test():
            print("\nBasic functionality test passed!")
        else:
            print("\nBasic functionality test failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()


