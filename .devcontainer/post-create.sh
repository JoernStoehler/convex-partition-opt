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