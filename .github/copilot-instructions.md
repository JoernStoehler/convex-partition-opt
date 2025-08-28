# Convex Partition Optimization

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup - NEVER CANCEL
- **CRITICAL: Install uv package manager first**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/home/vscode/.cargo/bin:$PATH"  # or ~/.cargo/bin for normal users
  ```
  **TIMING**: 2-3 minutes. NEVER CANCEL: Set timeout to 300+ seconds.

- **Create virtual environment and install dependencies**:
  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install -e ".[dev,docs]"
  ```
  **TIMING**: 3-5 minutes for full dependency installation. NEVER CANCEL: Set timeout to 600+ seconds.

- **Verify JAX installation**:
  ```bash
  python scripts/hello_jax.py
  ```
  **TIMING**: 10-15 seconds. This MUST complete successfully before proceeding.

### Build and Test Commands - NEVER CANCEL
- **Run linting and formatting**:
  ```bash
  source .venv/bin/activate
  ruff check src tests          # TIMING: 5-10 seconds
  ruff format src tests         # TIMING: 5-10 seconds
  ```
  **ALWAYS run these before committing**. Use `--check` flag to verify formatting without changes.

- **Type checking**:
  ```bash
  source .venv/bin/activate
  pyright src                   # TIMING: 15-30 seconds
  ```

- **Run tests**:
  ```bash
  source .venv/bin/activate
  pytest tests/                 # TIMING: 30-60 seconds for basic tests
  pytest tests/ --cov=src --cov-report=term-missing  # With coverage
  ```
  **NEVER CANCEL**: Test suite may take up to 5 minutes for full suite. Set timeout to 600+ seconds.

- **Build documentation**:
  ```bash
  source .venv/bin/activate
  mkdocs serve                  # TIMING: 5-10 seconds to start, runs continuously
  ```
  **Note**: Serves at http://localhost:8000. Stop with Ctrl+C.

### Pre-commit Setup
- **Install and setup pre-commit hooks**:
  ```bash
  source .venv/bin/activate
  pre-commit install           # TIMING: 10-15 seconds
  pre-commit run --all-files   # TIMING: 30-60 seconds first run
  ```

## Validation Scenarios

**CRITICAL**: After making any changes, ALWAYS test at least one complete scenario:

### Basic Geometry Validation
```bash
source .venv/bin/activate
python -c "
from convex_partition.reference.geometry import aspect_ratio
import numpy as np
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
ratio = aspect_ratio(square)
expected = np.sqrt(2)
print(f'Square aspect ratio: {ratio:.6f} (expected: {expected:.6f})')
assert abs(ratio - expected) < 1e-6, 'Square aspect ratio test failed'
print('✅ Basic geometry test passed')
"
```

### JAX Integration Test
```bash
source .venv/bin/activate
python scripts/hello_jax.py
```
**Expected output**: JAX version, square aspect ratio ≈ 1.414213, gradient calculation, triangle test.

### Test Suite Validation
```bash
source .venv/bin/activate
pytest tests/test_reference_geometry.py -v
```
**Expected**: All tests pass, coverage information displayed.

### Repository Structure Validation (Emergency Fallback)
If dependencies fail to install, run this comprehensive validation:
```bash
/usr/bin/python3 -c "
import os, sys
sys.path.insert(0, './src')

# Check directory structure
required = ['src/convex_partition', 'tests', 'scripts', 'docs', '.github']
for d in required: print(f'✅ {d}' if os.path.exists(d) else f'❌ {d}')

# Test basic functionality without external deps
from convex_partition.reference.geometry_simple import polygon_area, aspect_ratio
square = [[0, 0], [1, 0], [1, 1], [0, 1]]
area, ratio = polygon_area(square), aspect_ratio(square)
print(f'Area: {area}, Ratio: {ratio:.6f}')
assert abs(area - 1.0) < 1e-6 and abs(ratio - 1.414214) < 1e-6
print('✅ Basic validation passed')
"
```

## Key Locations and Navigation

### Source Code Structure
```
src/convex_partition/
├── __init__.py                    # Package initialization
├── reference/                     # NumPy-based reference implementations
│   ├── __init__.py
│   ├── geometry.py               # Core geometric primitives (CRITICAL)
│   ├── partition.py              # Partition data structures
│   └── scoring.py                # Aspect ratio calculations
└── efficient/                    # JAX-optimized implementations
    ├── __init__.py
    └── geometry.py               # JAX versions (must match reference)
```

### Test Organization
```
tests/
├── conftest.py                   # Shared fixtures (unit_square, tolerances)
├── test_reference_geometry.py    # Basic geometric operations
├── test_reference_partition.py   # Partition validation
├── test_paper_verification.py    # Reproduce paper's claimed results
├── test_efficient_consistency.py # Reference vs JAX consistency
├── test_hypothesis.py            # Property-based testing
└── artifacts/                    # Generated SVGs, CSVs during tests
```

### Frequently Used Commands
```bash
# Quick development cycle
source .venv/bin/activate
pytest tests/test_reference_geometry.py  # Test core geometry
ruff format src/ tests/                   # Format code
pyright src/                              # Type check

# Before committing changes
ruff check src tests                      # Lint
ruff format --check src tests             # Verify formatting
pytest tests/                             # Run all tests
pyright src/                              # Type check

# Documentation workflow
mkdocs serve                              # Preview docs at localhost:8000
```

## Repository Configuration Files

### Critical Files - NEVER modify without understanding
- `pyproject.toml`: Package metadata, dependencies, tool configuration
- `.pre-commit-config.yaml`: Pre-commit hooks (ruff, formatting, YAML checks)
- `mkdocs.yml`: Documentation site configuration
- `tests/conftest.py`: Shared test fixtures and configuration

### Development Dependencies
- **Core**: `jax[cpu]>=0.4.30`, `numpy>=1.26`, `matplotlib>=3.8`
- **Testing**: `pytest>=8.0`, `hypothesis>=6.100`, `pytest-cov>=5.0`
- **Linting**: `ruff>=0.6`, `pyright>=1.1.370`
- **Docs**: `mkdocs>=1.6`, `mkdocs-material>=9.5`

## Common Development Tasks

### Adding New Geometry Function
1. **Implement in reference first**: `src/convex_partition/reference/geometry.py`
2. **Add comprehensive tests**: `tests/test_reference_geometry.py`
3. **Validate with known examples**: Use paper's benchmark values
4. **Implement JAX version**: `src/convex_partition/efficient/geometry.py`
5. **Add consistency test**: Compare reference vs JAX implementations

### Debugging Numerical Issues
1. **Check for degeneracies**: Near-zero areas (< 1e-12), edge lengths (< 1e-6)
2. **Verify polygon orientation**: Should be counter-clockwise
3. **Use proper tolerances**: `rtol=1e-9, atol=1e-12` for reference comparisons
4. **Generate visual artifacts**: Always create SVG files for inspection

### Paper Benchmarks to Match
- **12-piece partition**: γ = 1.33964
- **21-piece partition**: γ = 1.32348  
- **37-piece partition**: γ = 1.31539
- **92-piece partition**: γ = 1.29950 (best known)

## CI/CD Pipeline Requirements

Every push/PR must pass (`.github/workflows/ci.yml`):
1. **Linting**: `ruff check` with no errors (5-10 seconds)
2. **Formatting**: `ruff format --check` unchanged (5-10 seconds)  
3. **Type checking**: `pyright` with strict mode (15-30 seconds)
4. **Tests**: All pytest tests passing (30-300 seconds depending on scope)
5. **Coverage**: Maintained >90% coverage
6. **Performance**: Key operations within expected time limits

## Network Dependencies and Limitations

**CRITICAL**: This project requires internet access for:
- Installing `uv` package manager from `astral.sh`
- Installing Python packages from PyPI
- If network access is limited:
  - Document which commands fail: "pip install fails due to network limitations"
  - Use system Python packages when available
  - Note timing for when network is restored

## Timeout Specifications - VALIDATED TIMINGS

**NEVER CANCEL these operations**:
- **Package installation**: 600+ seconds (10+ minutes) - *NOT tested due to network limits*
- **Python compilation**: <5 seconds - *VALIDATED: 0.034s for all source files*
- **Basic geometry tests**: <5 seconds - *VALIDATED: immediate response*
- **Repository structure validation**: <5 seconds - *VALIDATED: immediate response*
- **Full test suite**: 600+ seconds (10+ minutes) - *NOT tested due to dependency limits*
- **Documentation build**: 300+ seconds (5+ minutes) - *NOT tested due to dependency limits*
- **Type checking**: 120+ seconds (2+ minutes) - *NOT tested due to dependency limits*
- **JAX compilation (first run)**: 180+ seconds (3+ minutes) - *NOT tested due to dependency limits*

**Commands successfully validated in isolated environment**:
- ✅ `python -m py_compile` (syntax validation)
- ✅ Basic geometry calculations without external dependencies
- ✅ Repository structure validation
- ✅ Package imports and version checking
- ❌ JAX/numpy operations (requires network installation)
- ❌ pytest execution (requires installation)
- ❌ ruff linting (requires installation)
- ❌ pyright type checking (requires installation)

## Mathematical Context

This project optimizes convex partitions of the unit square to minimize the maximum aspect ratio (circumradius/inradius) across all polygons. Key concepts:
- **Aspect ratio γ**: circumradius/inradius (want close to 1.0)
- **Partition score**: max aspect ratio across all pieces (minimize this)
- **Current best**: γ = 1.29950 with 92 pieces (from 2003 paper)
- **Goal**: Improve bounds using gradient-based optimization

## Implementation Philosophy

**Three-tier approach**:
1. **Reference** (`reference/`): NumPy-based, prioritizes clarity and correctness
2. **Efficient** (`efficient/`): JAX-based for autodiff, must match reference to 1e-6
3. **Graphless** (`optimize/`): Dynamic topology optimization

**Always validate against reference implementation first**.

## Emergency Procedures - Network/Dependency Issues

**If pip/uv installation fails due to network limitations**:
1. **Document the failure**: "pip install fails due to network limitations"
2. **Use system Python for basic validation**:
   ```bash
   /usr/bin/python3 -m py_compile src/convex_partition/*.py  # Test syntax
   ```
3. **Create minimal test without external dependencies**:
   ```bash
   /usr/bin/python3 -c "
   import sys; sys.path.insert(0, './src')
   from convex_partition.reference.geometry_simple import polygon_area, aspect_ratio
   import math
   square = [[0, 0], [1, 0], [1, 1], [0, 1]]
   print(f'Area: {polygon_area(square)}, Ratio: {aspect_ratio(square):.6f}')
   "
   ```

**Expected output for minimal validation**: 
```
Area: 1.0, Ratio: 1.414214
```

## Common Error Patterns

1. **"JAX not found"**: 
   - First: Check if virtual environment is activated
   - Then: Run `python scripts/hello_jax.py` to verify installation
   - Fallback: Use system Python with `geometry_simple.py` for basic validation

2. **"Module not found"**: 
   - Ensure virtual environment is activated: `source .venv/bin/activate`
   - Check Python path: `python -c "import sys; print(sys.path)"`
   - Verify installation: `pip list | grep convex-partition`

3. **"Network timeouts during installation"**:
   - Document exact error and retry time
   - Use system packages when available
   - Set longer timeouts: `pip install --timeout=300`

4. **"Tests failing"**: 
   - **Precision errors** (acceptable): differences < 1e-6 in numerical calculations
   - **Logic errors** (fix required): wrong shapes, missing vertices, incorrect topology
   - Run individual test files: `pytest tests/test_reference_geometry.py -v`

5. **"Type errors"**: 
   - Use `# type: ignore` only for external library stubs
   - Add proper type annotations for new functions
   - Run `pyright src/` to see all type issues

6. **"Build timeout errors"**:
   - NEVER CANCEL - always wait for completion
   - JAX first compilation can take 3+ minutes
   - Set explicit timeouts: `--timeout=600` for bash commands