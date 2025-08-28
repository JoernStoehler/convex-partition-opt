# Project Infrastructure Specification

## Overview
Repository for computational geometry research on optimal convex partitions of regular polygons, particularly squares. Uses JAX for differentiable optimization with comprehensive testing and documentation infrastructure.

## Repository Structure

```
convex-partition-opt/
├── .devcontainer/
│   └── devcontainer.json          # GitHub Codespaces configuration
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # Test + typecheck + lint on push
│   │   └── docs.yml               # Build & deploy docs to GitHub Pages
│   └── dependabot.yml             # Auto-update dependencies
├── .vscode/
│   ├── settings.json              # Project-specific VSCode settings
│   └── extensions.json            # Recommended extensions
├── src/
│   └── convex_partition/
│       ├── __init__.py
│       ├── reference/             # Trusted, clear implementations
│       │   ├── __init__.py
│       │   ├── geometry.py        # Basic geometric primitives
│       │   ├── partition.py       # Partition data structures
│       │   └── scoring.py         # Aspect ratio calculations
│       └── efficient/             # JAX-optimized implementations
│           └── __init__.py        # Empty initially
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures and test config
│   ├── test_reference_geometry.py # Basic geometric primitives
│   ├── test_reference_partition.py # Partition validation
│   ├── test_paper_verification.py # Verify paper's claimed results
│   ├── test_efficient_consistency.py # Compare reference vs JAX
│   ├── test_hypothesis.py         # Property-based testing
│   └── artifacts/                 # Generated test outputs
│       └── .gitkeep
├── docs/
│   ├── index.md                   # Project overview
│   ├── paper_summary.md           # Summary of CCCG 2003 paper
│   └── results/                   # Generated figures & results
├── notebooks/
│   └── exploration.ipynb          # Jupyter notebooks for experiments
├── scripts/
│   └── hello_jax.py              # Demo JAX script for setup validation
├── .gitignore                     # Python, JAX, IDE exclusions
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
├── pyproject.toml                 # Project metadata & tool configs
├── README.md                      # Public-facing documentation
├── CLAUDE.md                      # Claude Code onboarding guide
├── spec.md                        # This file
└── mkdocs.yml                     # Documentation site configuration
```

## Core Dependencies

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
pip-log.txt
.pytest_cache/
.coverage
htmlcov/
.hypothesis/
*.egg-info/

# JAX / NumPy
*.npy
*.npz

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Project specific
results/*.svg
results/*.png
results/*.gif
tests/artifacts/*
!tests/artifacts/.gitkeep
site/
dist/
build/
*.log

# MkDocs
site/

# UV
.uv/
uv.lock
```

### .vscode/settings.json
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "editor.tabSize": 2,
  "editor.insertSpaces": true,
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "editor.bracketPairColorization.enabled": true,
  "editor.guides.bracketPairs": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit",
      "source.fixAll": "explicit"
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.autoImportCompletions": true,
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/*.egg-info": true,
    ".ruff_cache": true
  }
}
```

### pyproject.toml
```toml
[project]
name = "convex-partition"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "jax[cpu]>=0.4.30",  # Use jax[cuda12] for GPU support
    "jaxtyping>=0.2.25",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "svgwrite>=1.4",
    "beartype>=0.18",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "hypothesis>=6.100",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.5",  # Parallel test execution
    "ruff>=0.6",
    "pyright>=1.1.370",
    "ipykernel>=6.29",    # For notebooks
]
docs = [
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.25",
    "mkdocs-jupyter>=0.24",  # Include notebooks in docs
]
ml = [
    "flax>=0.8",          # Neural networks in JAX
    "optax>=0.2",         # Optimizers for JAX
    "orbax-checkpoint>=0.5",  # Checkpointing
    "tensorboardX>=2.6",  # Logging
    "pandas>=2.2",        # Data analysis
    "seaborn>=0.13",      # Statistical plots
]
gpu = [
    "jax[cuda12]>=0.4.30",  # GPU support via CUDA 12
    # Note: May need to set JAX_CUDA_VERSION=12 environment variable
]

[tool.ruff]
line-length = 100
indent-width = 2
target-version = "py311"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "strict"
reportMissingImports = true
reportMissingTypeStubs = false  # JAX stubs are incomplete

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers -v"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Devcontainer Configuration

### .devcontainer/devcontainer.json
```json
{
  "name": "Convex Partition Optimization",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers-contrib/features/mkdocs:2": {}
  },
  "postCreateCommand": ".devcontainer/post-create.sh",
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/workspaces/convex-partition-opt/.venv/bin/python",
        "python.terminal.activateEnvironment": true,
        "editor.tabSize": 2,
        "editor.insertSpaces": true,
        "editor.formatOnSave": true,
        "editor.rulers": [100],
        "[python]": {
          "editor.defaultFormatter": "charliermarsh.ruff"
        }
      },
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-toolsai.jupyter",
        "mechatroner.rainbow-csv",
        "PKief.material-icon-theme",
        "github.copilot",
        "ms-python.debugpy",
        "tamasfe.even-better-toml"
      ]
    }
  },
  "forwardPorts": [8000],
  "remoteUser": "vscode"
}
```

### .devcontainer/post-create.sh
```bash
#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting development environment setup..."

# Install uv package manager
echo "📦 Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/home/vscode/.cargo/bin:$PATH"

# Create virtual environment and install dependencies
echo "🐍 Creating Python environment..."
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,docs]"

# Verify JAX installation
echo "🔬 Verifying JAX installation..."
python -c "import jax; print(f'JAX {jax.__version__} installed successfully')"

# Install pre-commit hooks
echo "🔗 Installing pre-commit hooks..."
pre-commit install

# Install Claude Code (if available in environment)
if command -v npm &> /dev/null; then
  echo "🤖 Attempting Claude Code installation..."
  npm install -g @anthropic/claude-code 2>/dev/null || echo "⚠️  Claude Code requires manual setup"
fi

# Create directories
echo "📁 Creating project directories..."
mkdir -p tests/artifacts results notebooks/figures

# Run initial tests
echo "🧪 Running hello world test..."
python scripts/hello_jax.py

echo "✅ Development environment setup complete!"
echo "📝 Next steps:"
echo "  1. Run 'source .venv/bin/activate' to activate Python environment"
echo "  2. Run 'pytest tests/' to verify test infrastructure"
echo "  3. Run 'mkdocs serve' to preview documentation"
```

# Make script executable (add to setup instructions)
# chmod +x .devcontainer/post-create.sh

## Pre-commit Configuration

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff          # Linting
        args: [--fix]
      - id: ruff-format   # Formatting

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
        additional_dependencies: [jaxtyping, beartype]
        args: [--ignore-missing-imports]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=1000]
```

## GitHub Actions

### .github/workflows/ci.yml
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install uv
      uses: astral-sh/setup-uv@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        uv venv
        uv pip install -e ".[dev]"
    
    - name: Run ruff
      run: |
        source .venv/bin/activate
        ruff check src tests
        ruff format --check src tests
    
    - name: Run pyright
      run: |
        source .venv/bin/activate
        pyright src
    
    - name: Run tests
      run: |
        source .venv/bin/activate
        pytest tests --cov=src --cov-report=term-missing
```

### .github/workflows/docs.yml
```yaml
name: Deploy docs

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install uv
      uses: astral-sh/setup-uv@v3
    
    - name: Install dependencies
      run: |
        uv venv
        uv pip install -e ".[docs]"
    
    - name: Build docs
      run: |
        source .venv/bin/activate
        mkdocs build
    
    - name: Upload artifact
      uses: actions/upload-pages-artifact@v3
      with:
        path: ./site
    
    - name: Deploy to GitHub Pages
      uses: actions/deploy-pages@v4
```

## Documentation Configuration

### mkdocs.yml
```yaml
site_name: Convex Partition Optimization
site_description: Optimal convex partitions of regular polygons
site_url: https://username.github.io/convex-partition-opt

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - content.code.annotate
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
  - mkdocs-jupyter:
      include_source: true

nav:
  - Home: index.md
  - Paper Summary: paper_summary.md
  - API Reference:
      - Reference Implementation: api/reference.md
      - Efficient Implementation: api/efficient.md
  - Notebooks:
      - Exploration: notebooks/exploration.ipynb

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.arithmatex:
      generic: true
  - admonition
  - codehilite

extra_javascript:
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
```

## Sample Files

### tests/test_reference_geometry.py (sample)
```python
"""Test basic geometric operations."""
import numpy as np
import pytest
from convex_partition.reference.geometry import (
    aspect_ratio, circumcircle, incircle_convex_polygon, 
    is_convex, polygon_area
)

def test_square_aspect_ratio():
  """Test aspect ratio of unit square is √2."""
  square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
  ratio = aspect_ratio(square)
  expected = np.sqrt(2)
  assert np.abs(ratio - expected) < 1e-10, f"Expected {expected}, got {ratio}"

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

@pytest.mark.parametrize("n_sides", [3, 4, 5, 6, 8])
def test_regular_polygons(n_sides):
  """Test aspect ratios of regular polygons."""
  angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
  vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)
  ratio = aspect_ratio(vertices)
  # All regular polygons should have aspect ratio < 2.1
  assert 1.0 <= ratio <= 2.1, f"Unexpected ratio {ratio} for {n_sides}-gon"
```

### scripts/hello_jax.py
```python
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
```

### CLAUDE.md (stub)
```markdown
# Claude Code Onboarding Guide

## Project Overview
This project implements computational methods to find optimal convex partitions of regular polygons (focusing on squares) that minimize the aspect ratio (circumradius/inradius). We aim to improve upon the 2003 CCCG paper's bounds of [1.28868, 1.29950].

## Key Mathematical Concepts
- **Aspect Ratio γ**: ratio of circumcircle to incircle radii (want close to 1)
- **Convex Partition**: decomposition into non-overlapping convex polygons
- **Score**: max aspect ratio across all polygons in partition (minimize this)
- **Unit Square**: [0,1]² with fixed corners at (0,0), (1,0), (1,1), (0,1)

## Three Implementation Levels
1. **Reference** (`src/convex_partition/reference/`): 
   - Uses NumPy, prioritizes clarity
   - Extensive validation and assertions
   - Our source of truth for correctness
   
2. **Efficient** (`src/convex_partition/efficient/`):
   - Uses JAX for autodiff
   - Batched operations with vmap
   - Must match reference to 1e-6
   
3. **Graphless** (in `optimize/`):
   - Optimizes vertex positions without fixed topology
   - Dynamically selects best graph structure

## Development Workflow
```bash
# Always work in virtual environment
source .venv/bin/activate

# Before committing
pytest tests/                    # Run all tests
pyright src/                     # Type checking
ruff format src/ tests/          # Format code
ruff check src/ tests/           # Lint code

# Documentation
mkdocs serve                     # Preview at localhost:8000

# Running optimizations
python -m convex_partition.optimize.fixed_topology  # Optimize paper's partition
python scripts/analyze_patterns.py                  # Extract patterns
```

## Testing Philosophy
- **Reference tests**: Edge cases, visual validation via SVG
- **Efficient tests**: Consistency with reference, gradient checks
- **Hypothesis tests**: Property-based testing of invariants
- **Visual tests**: Always generate SVGs for inspection

## Common Tasks

### Adding a new geometry function
1. Implement in `reference/geometry.py` with docstring
2. Add comprehensive tests in `tests/test_reference_geometry.py`
3. Once validated, implement efficient version in `efficient/geometry.py`
4. Add consistency test comparing both implementations

### Debugging numerical issues
1. Check for near-zero denominators (area, edge lengths)
2. Verify polygon orientation (should be CCW)
3. Use `np.allclose` not `==` for float comparisons
4. For gradients: check if you're at constraint boundaries

### Understanding partition format
```python
partition = ConvexPartition(
  vertices=Vertices(
    corners=np.array([[0,0], [1,0], [1,1], [0,1]]),  # Fixed
    edge_points=np.array([[0.5, 0], [0.5, 1]]),      # On edges
    interior_points=np.array([[0.5, 0.5]])           # Free
  ),
  polygons=[
    [0, 1, 6],  # Triangle using vertices 0, 1, 6
    [1, 2, 6],  # Adjacent triangle
    # ... more polygons covering the square
  ]
)
```

## Key Algorithms

### Incircle calculation (reference)
```python
# For convex polygon: find smallest distance from center to any edge
center = vertices.mean(axis=0)
for i in range(len(vertices)):
  edge = vertices[(i+1)%n] - vertices[i]
  to_center = center - vertices[i]
  # Distance from center to edge (point-to-line formula)
  dist = abs(np.cross(edge, to_center)) / np.linalg.norm(edge)
  min_dist = min(min_dist, dist)
```

### Graph enumeration strategy
1. Start with Delaunay triangulation
2. Try merging adjacent triangles if result stays convex
3. Prune if partial score exceeds best complete score
4. Cache convexity checks for efficiency

## Critical Numerical Thresholds
- **Degeneracy threshold**: area < 1e-12 → ignore polygon
- **Float comparison**: use rtol=1e-9, atol=1e-12
- **Gradient clipping**: clip to [-10, 10] if exploding
- **Minimum edge length**: 1e-6 to avoid numerical issues

## Paper Benchmarks to Match
- 12-piece partition: γ = 1.33964
- 21-piece partition: γ = 1.32348  
- 37-piece partition: γ = 1.31539
- 92-piece partition: γ = 1.29950 (best known)

## Questions for Human Review
When unsure about:
1. Numerical stability issues → Show error and simplified test case
2. Optimization not converging → Plot loss curve, check gradients
3. Unexpected partition topology → Generate SVG for visual inspection
4. Test failures → Check if it's precision (ok) or logic (not ok)

## External Resources
- Original paper: `/docs/CCCG2003_Damian_ORourke.pdf`
- JAX documentation: https://jax.readthedocs.io/
- Computational Geometry: de Berg et al. textbook
```

## Setup Instructions for DevOps

1. Create new GitHub repository with this structure
2. Enable GitHub Pages from Actions in repository settings
3. Set up branch protection rules for `main` requiring CI checks
4. Create initial commit with all config files
5. Verify GitHub Codespaces launches correctly
6. Run `scripts/hello_jax.py` to verify JAX installation
7. Run `pytest tests/` with a simple test to verify test infrastructure
8. Deploy initial docs to GitHub Pages

## Notes
- Avoid custom Docker images to maintain fast Codespaces startup
- The `mcr.microsoft.com/devcontainers/python:3.11` image is pre-cached by GitHub
- Claude Code installation may require manual intervention depending on availability
- Consider adding GPU support later via `jax[cuda12]` for larger optimizations
