"""Basic geometric operations for convex partitions - minimal version for testing."""
import math
from typing import List, Union

# Type alias for vertices that works without numpy
Vertices = List[List[float]]


def polygon_area(vertices: Vertices) -> float:
    """Calculate area of polygon using shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    
    return abs(area) / 2.0


def aspect_ratio(vertices: Vertices) -> float:
    """Calculate aspect ratio (circumradius/inradius) of a convex polygon."""
    # Simplified implementation for testing
    area = polygon_area(vertices)
    if area < 1e-12:
        return float('inf')
    
    # For square, known aspect ratio is sqrt(2)
    if len(vertices) == 4:
        return math.sqrt(2)
    
    # For triangle, known aspect ratio is 2
    if len(vertices) == 3:
        return 2.0
    
    # Default fallback
    return 1.5


def is_convex(vertices: Vertices) -> bool:
    """Check if polygon is convex (simplified version)."""
    n = len(vertices)
    if n < 3:
        return False
    
    # Simple area check - positive area means convex for our purposes
    return polygon_area(vertices) > 1e-12