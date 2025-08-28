"""Test basic geometric operations."""
import numpy as np
import pytest
from convex_partition.reference.geometry import aspect_ratio, polygon_area


def test_square_aspect_ratio(unit_square):
    """Test aspect ratio of unit square is √2."""
    ratio = aspect_ratio(unit_square)
    expected = np.sqrt(2)
    assert np.abs(ratio - expected) < 1e-10, f"Expected {expected}, got {ratio}"


def test_square_area(unit_square):
    """Test area of unit square is 1."""
    area = polygon_area(unit_square)
    assert np.abs(area - 1.0) < 1e-10, f"Expected 1.0, got {area}"


def test_equilateral_triangle_aspect_ratio():
    """Test aspect ratio of equilateral triangle is 2."""
    triangle = np.array([
        [0, 0],
        [1, 0], 
        [0.5, np.sqrt(3)/2]
    ], dtype=np.float64)
    ratio = aspect_ratio(triangle)
    assert np.abs(ratio - 2.0) < 1e-10, f"Expected 2.0, got {ratio}"


def test_degenerate_polygon():
    """Test handling of nearly collinear vertices."""
    degenerate = np.array([[0, 0], [1, 0], [0.5, 1e-15]], dtype=np.float64)
    area = polygon_area(degenerate)
    assert area < 1e-12, "Should detect near-zero area"


@pytest.mark.parametrize("n_sides", [3, 4, 5, 6])
def test_polygon_areas(n_sides):
    """Test area calculation for regular polygons."""
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    area = polygon_area(vertices)
    # Unit circle has area π, inscribed polygon should be smaller
    assert 0 < area < np.pi, f"Unexpected area {area} for {n_sides}-gon"