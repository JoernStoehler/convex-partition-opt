"""
Tests for C integration functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_c_integration_simple():
    """Test basic C integration without numpy dependencies."""
    from convex_partition.c_interface_simple import test_c_integration_simple
    
    result = test_c_integration_simple()
    assert result is True, "C integration test should pass"

def test_c_library_loading():
    """Test that C library can be loaded and basic functions work."""
    from convex_partition.c_interface_simple import VertexCloud, Polygon
    import ctypes
    from pathlib import Path
    
    # Find library
    lib_paths = [
        Path(__file__).parent.parent / "convex_partition" / "lib" / "libconvex_partition.so",
        Path(__file__).parent.parent / "src" / "c_extensions" / "libconvex_partition.so",
    ]
    
    lib_path = None
    for path in lib_paths:
        if path.exists():
            lib_path = path
            break
    
    assert lib_path is not None, f"C library not found in {lib_paths}"
    
    # Load library
    lib = ctypes.CDLL(str(lib_path))
    
    # Set up function signatures
    lib.init_vertex_cloud.argtypes = [ctypes.POINTER(VertexCloud)]
    lib.init_vertex_cloud.restype = None
    
    lib.loss_poly.argtypes = [ctypes.POINTER(Polygon), ctypes.POINTER(VertexCloud)]
    lib.loss_poly.restype = ctypes.c_double
    
    # Test initialization
    cloud = VertexCloud()
    lib.init_vertex_cloud(ctypes.byref(cloud))
    
    assert cloud.n_corner == 4
    assert cloud.n == 4
    assert abs(cloud.x[0] - 0.0) < 1e-10
    assert abs(cloud.y[0] - 0.0) < 1e-10
    assert abs(cloud.x[2] - 1.0) < 1e-10
    assert abs(cloud.y[2] - 1.0) < 1e-10
    
    # Test loss calculation
    poly = Polygon()
    lib.init_polygon(ctypes.byref(poly))
    
    loss = lib.loss_poly(ctypes.byref(poly), ctypes.byref(cloud))
    expected_loss = 1.4142135623730950488  # sqrt(2) for unit square
    assert abs(loss - expected_loss) < 1e-10, f"Expected {expected_loss}, got {loss}"

if __name__ == "__main__":
    test_c_integration_simple()
    test_c_library_loading()
    print("✅ All C integration tests passed!")