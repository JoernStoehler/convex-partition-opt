#!/usr/bin/env python3
"""
Hello world script for JAX setup validation.

This script verifies that JAX is properly installed and can perform
basic operations including differentiation, which is essential for
the convex partition optimization project.
"""

import jax
import jax.numpy as jnp
from jax import grad


def hello_jax():
    """Demonstrate basic JAX functionality."""
    print(f"🔬 JAX version: {jax.__version__}")
    print(f"🖥️  JAX backend: {jax.default_backend()}")
    print(f"📊 Available devices: {jax.devices()}")
    
    # Test basic array operations
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    dot_product = jnp.dot(x, y)
    print(f"✨ Dot product of {x} and {y}: {dot_product}")
    
    # Test differentiation (crucial for optimization)
    def f(x):
        return x ** 2 + 3 * x + 1
    
    df_dx = grad(f)
    x_test = 2.0
    derivative = df_dx(x_test)
    print(f"🧮 Derivative of f(x) = x² + 3x + 1 at x={x_test}: {derivative}")
    print(f"   Expected: {2 * x_test + 3}")
    
    # Test matrix operations (needed for geometry)
    A = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([5, 6])
    result = A @ b
    print(f"🔢 Matrix multiplication A @ b = {result}")
    
    print("✅ JAX is working correctly!")
    return True


if __name__ == "__main__":
    hello_jax()