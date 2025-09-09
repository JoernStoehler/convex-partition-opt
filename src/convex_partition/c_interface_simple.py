"""
Simple Python interface to C convex partition extensions (without numpy dependency).

This module provides basic Python bindings for testing C integration.
"""

import ctypes
from pathlib import Path

# Constants matching C definitions
MAX_VERTICES = 1000
MAX_POLYGONS = 100
MAX_POLYGON_SIZE = 20

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

def test_c_integration_simple():
    """Test basic C integration functionality without numpy."""
    print("🧪 Testing C integration (simple)...")
    
    try:
        # Load library
        lib_paths = [
            Path(__file__).parent / "lib" / "libconvex_partition.so",
            Path(__file__).parent.parent.parent / "convex_partition" / "lib" / "libconvex_partition.so",
            Path("./src/c_extensions/libconvex_partition.so"),
        ]
        
        lib_path = None
        for path in lib_paths:
            if path.exists():
                lib_path = path
                break
        
        if lib_path is None:
            print(f"❌ Library not found. Tried: {[str(p) for p in lib_paths]}")
            return False
            
        lib = ctypes.CDLL(str(lib_path))
        print(f"✅ Loaded C library from {lib_path}")
        
        # Set up function signatures
        lib.init_vertex_cloud.argtypes = [ctypes.POINTER(VertexCloud)]
        lib.init_vertex_cloud.restype = None
        
        lib.init_polygon.argtypes = [ctypes.POINTER(Polygon)]
        lib.init_polygon.restype = None
        
        lib.loss_poly.argtypes = [ctypes.POINTER(Polygon), ctypes.POINTER(VertexCloud)]
        lib.loss_poly.restype = ctypes.c_double
        
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
        
        print("🎉 Simple C integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ C integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_c_integration_simple()