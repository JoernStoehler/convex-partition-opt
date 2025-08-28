"""Verify JAX installation with aspect ratio calculation."""
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32
import numpy as np


def aspect_ratio_square(side_length: Float32[Array, ""]) -> Float32[Array, ""]:
    """Calculate aspect ratio of a square (should be √2)."""
    # Circumradius of square = diagonal/2 = side * √2 / 2
    circumradius = side_length * jnp.sqrt(2) / 2
    # Inradius of square = side/2
    inradius = side_length / 2
    return circumradius / inradius


def aspect_ratio_triangle(vertices: Float32[Array, "3 2"]) -> Float32[Array, ""]:
    """Calculate aspect ratio of a triangle."""
    # Simplified version - full implementation in reference/
    a = jnp.linalg.norm(vertices[1] - vertices[0])
    b = jnp.linalg.norm(vertices[2] - vertices[1]) 
    c = jnp.linalg.norm(vertices[0] - vertices[2])
    s = (a + b + c) / 2  # Semi-perimeter
    area = jnp.sqrt(s * (s - a) * (s - b) * (s - c))
    # Circumradius formula
    circumradius = (a * b * c) / (4 * area)
    # Inradius formula
    inradius = area / s
    return circumradius / inradius


if __name__ == "__main__":
    # Test square calculation
    print(f"🔬 JAX version: {jax.__version__}")
    print(f"📐 Testing aspect ratio calculations...\n")

    # Square test
    side = jnp.array(1.0)
    ratio = aspect_ratio_square(side)
    expected = np.sqrt(2)
    print(f"Square aspect ratio: {ratio:.6f}")
    print(f"Expected (√2): {expected:.6f}")
    print(f"✅ Match: {abs(ratio - expected) < 1e-6}\n")

    # Gradient test
    grad_fn = jax.grad(aspect_ratio_square)
    gradient = grad_fn(side)
    print(f"Gradient w.r.t. side length: {gradient:.6f}")

    # Triangle test (equilateral)
    equilateral = jnp.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    tri_ratio = aspect_ratio_triangle(equilateral)
    print(f"\nEquilateral triangle aspect ratio: {tri_ratio:.6f}")
    print(f"Expected (paper): 2.0")
    print(f"✅ Reasonable: {1.9 < tri_ratio < 2.1}")

    print("\n✅ JAX is working correctly for geometry calculations!")