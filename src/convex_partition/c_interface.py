"""
Python interface to C convex partition extensions.

This module provides Python bindings for high-performance C implementations
of convex partition optimization algorithms.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

# Constants matching C definitions
MAX_VERTICES = 1000
MAX_POLYGONS = 100
MAX_POLYGON_SIZE = 20

# Load the shared library
def _load_c_library():
    """Load the C shared library."""
    # Try different possible locations
    lib_paths = [
        Path(__file__).parent / "lib" / "libconvex_partition.so",
        Path(__file__).parent.parent / "c_extensions" / "libconvex_partition.so",
        Path("src/c_extensions/libconvex_partition.so"),
        Path("libconvex_partition.so"),
    ]
    
    for lib_path in lib_paths:
        if lib_path.exists():
            return ctypes.CDLL(str(lib_path))
    
    raise RuntimeError(
        f"Could not find libconvex_partition.so. Tried: {[str(p) for p in lib_paths]}\n"
        "Run 'make -C src/c_extensions lib' to build the library."
    )

# Data structure definitions matching C structs
class VertexCloud(ctypes.Structure):
    """Python wrapper for vertex_cloud C struct."""
    _fields_ = [
        ("n_inner", ctypes.c_int),
        ("n_edge", ctypes.c_int * 4),
        ("n_corner", ctypes.c_int),
        ("n", ctypes.c_int),
        ("x", ctypes.c_double * MAX_VERTICES),
        ("y", ctypes.c_double * MAX_VERTICES),
    ]

class Polygon(ctypes.Structure):
    """Python wrapper for polygon C struct."""
    _fields_ = [
        ("s", ctypes.c_int),
        ("vertices", ctypes.c_int * MAX_POLYGON_SIZE),
    ]

class Partition(ctypes.Structure):
    """Python wrapper for partition C struct."""
    _fields_ = [
        ("n_poly", ctypes.c_int),
        ("s_max", ctypes.c_int),
        ("s", ctypes.c_int * MAX_POLYGONS),
        ("pi", (ctypes.c_int * MAX_POLYGON_SIZE) * MAX_POLYGONS),
    ]

class Gradient(ctypes.Structure):
    """Python wrapper for gradient C struct."""
    _fields_ = [
        ("dxy", (ctypes.c_double * 2) * MAX_POLYGON_SIZE),
    ]

# Global library reference
_lib = None

def get_lib():
    """Get the C library, loading it if necessary."""
    global _lib
    if _lib is None:
        _lib = _load_c_library()
        _setup_function_signatures()
    return _lib

def _setup_function_signatures():
    """Set up ctypes function signatures for the C library."""
    lib = _lib
    
    # Initialization functions
    lib.init_vertex_cloud.argtypes = [ctypes.POINTER(VertexCloud)]
    lib.init_vertex_cloud.restype = None
    
    lib.init_partition.argtypes = [ctypes.POINTER(Partition)]
    lib.init_partition.restype = None
    
    lib.init_polygon.argtypes = [ctypes.POINTER(Polygon)]
    lib.init_polygon.restype = None
    
    # Validation functions
    lib.validate_vertex_cloud.argtypes = [ctypes.POINTER(VertexCloud)]
    lib.validate_vertex_cloud.restype = ctypes.c_int
    
    lib.validate_partition.argtypes = [ctypes.POINTER(Partition), ctypes.POINTER(VertexCloud)]
    lib.validate_partition.restype = ctypes.c_int
    
    lib.validate_polygon.argtypes = [ctypes.POINTER(Polygon), ctypes.POINTER(VertexCloud)]
    lib.validate_polygon.restype = ctypes.c_int
    
    # Loss functions
    lib.loss_poly.argtypes = [ctypes.POINTER(Polygon), ctypes.POINTER(VertexCloud)]
    lib.loss_poly.restype = ctypes.c_double
    
    lib.loss_part.argtypes = [ctypes.POINTER(Partition), ctypes.POINTER(Polygon)]
    lib.loss_part.restype = ctypes.c_double
    
    lib.loss_vc.argtypes = [ctypes.POINTER(VertexCloud), ctypes.POINTER(Polygon), 
                           ctypes.POINTER(Partition)]
    lib.loss_vc.restype = ctypes.c_double
    
    lib.dloss_poly.argtypes = [ctypes.POINTER(Polygon), ctypes.POINTER(VertexCloud),
                              ctypes.POINTER(Gradient)]
    lib.dloss_poly.restype = ctypes.c_double

# High-level Python interface

def create_vertex_cloud(x: np.ndarray, y: np.ndarray, 
                       n_edge: Optional[List[int]] = None,
                       n_inner: Optional[int] = None) -> VertexCloud:
    """
    Create a vertex cloud from numpy arrays.
    
    Args:
        x: X coordinates of vertices
        y: Y coordinates of vertices  
        n_edge: Number of edge vertices for each edge [x=0, y=0, x=1, y=1]
        n_inner: Number of inner vertices
        
    Returns:
        VertexCloud structure
    """
    lib = get_lib()
    cloud = VertexCloud()
    lib.init_vertex_cloud(ctypes.byref(cloud))
    
    if len(x) != len(y):
        raise ValueError("x and y arrays must have same length")
    if len(x) > MAX_VERTICES:
        raise ValueError(f"Too many vertices: {len(x)} > {MAX_VERTICES}")
    
    # Copy coordinates
    cloud.n = len(x)
    for i in range(len(x)):
        cloud.x[i] = float(x[i])
        cloud.y[i] = float(y[i])
    
    # Set edge and inner counts if provided
    if n_edge is not None:
        if len(n_edge) != 4:
            raise ValueError("n_edge must have exactly 4 elements")
        for i in range(4):
            cloud.n_edge[i] = n_edge[i]
    
    if n_inner is not None:
        cloud.n_inner = n_inner
    
    return cloud

def create_polygon(vertices: List[int]) -> Polygon:
    """Create a polygon from vertex indices."""
    if len(vertices) > MAX_POLYGON_SIZE:
        raise ValueError(f"Too many vertices: {len(vertices)} > {MAX_POLYGON_SIZE}")
    
    lib = get_lib()
    poly = Polygon()
    lib.init_polygon(ctypes.byref(poly))
    
    poly.s = len(vertices)
    for i, v in enumerate(vertices):
        poly.vertices[i] = v
        
    return poly

def vertex_cloud_to_numpy(cloud: VertexCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Convert VertexCloud to numpy arrays."""
    x = np.array([cloud.x[i] for i in range(cloud.n)])
    y = np.array([cloud.y[i] for i in range(cloud.n)])
    return x, y

def polygon_loss(vertices: List[int], x: np.ndarray, y: np.ndarray) -> float:
    """Calculate loss for a single polygon."""
    lib = get_lib()
    cloud = create_vertex_cloud(x, y)
    poly = create_polygon(vertices)
    
    return lib.loss_poly(ctypes.byref(poly), ctypes.byref(cloud))

def polygon_gradient(vertices: List[int], x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate loss and gradient for a polygon."""
    lib = get_lib()
    cloud = create_vertex_cloud(x, y)
    poly = create_polygon(vertices)
    grad = Gradient()
    
    loss = lib.dloss_poly(ctypes.byref(poly), ctypes.byref(cloud), ctypes.byref(grad))
    
    # Extract gradient as numpy array
    gradient_array = np.zeros((len(vertices), 2))
    for i in range(len(vertices)):
        gradient_array[i, 0] = grad.dxy[i][0]
        gradient_array[i, 1] = grad.dxy[i][1]
    
    return loss, gradient_array

def test_c_integration():
    """Test basic C integration functionality."""
    print("🧪 Testing C integration...")
    
    try:
        lib = get_lib()
        print(f"✅ Loaded C library successfully")
        
        # Test basic structures
        cloud = VertexCloud()
        lib.init_vertex_cloud(ctypes.byref(cloud))
        assert cloud.n_corner == 4
        assert cloud.n == 4
        print(f"✅ VertexCloud initialization: n={cloud.n}")
        
        # Test loss calculation
        poly = Polygon()
        lib.init_polygon(ctypes.byref(poly))
        loss = lib.loss_poly(ctypes.byref(poly), ctypes.byref(cloud))
        print(f"✅ Loss calculation: {loss:.6f}")
        
        # Test high-level interface
        x = np.array([0.0, 1.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        vertices = [0, 1, 2, 3]
        
        poly_loss = polygon_loss(vertices, x, y)
        print(f"✅ High-level polygon loss: {poly_loss:.6f}")
        
        loss_grad, gradient = polygon_gradient(vertices, x, y)
        print(f"✅ Gradient computation: loss={loss_grad:.6f}, grad_shape={gradient.shape}")
        
        print("🎉 C integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ C integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_c_integration()