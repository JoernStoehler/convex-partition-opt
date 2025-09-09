#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting development environment setup..."

# Install essential development tools
echo "🔧 Installing development tools..."
sudo apt-get update -qq
sudo apt-get install -y build-essential gcc python3-dev python3-pip python3-venv

# Create Python virtual environment and install dependencies in background
echo "🐍 Setting up Python environment (running in background)..."
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies first (faster)
pip install --upgrade pip
pip install numpy>=1.26 matplotlib>=3.8 pytest>=8.0

echo "📦 Installing remaining Python dependencies in background..."
nohup bash -c "source .venv/bin/activate && pip install -e '.[dev,docs]' && echo 'Python deps installed' > .setup_complete" &

# Compile C extensions
echo "🔨 Compiling C extensions..."
make -C src/c_extensions clean all

# Create directories
echo "📁 Creating project directories..."
mkdir -p tests/artifacts results notebooks/figures

# Test C compilation
echo "🧪 Testing C integration..."
python -c "
import sys; sys.path.insert(0, './src')
from convex_partition.c_interface import test_c_integration
test_c_integration()
print('✅ C integration test passed')
"

echo "✅ Core development environment ready!"
echo "📝 Python dependencies installing in background..."
echo "📝 Quick start:"
echo "  1. Run 'source .venv/bin/activate' to activate Python environment"
echo "  2. Run 'make -C src/c_extensions test' to test C code"
echo "  3. Run 'python -c \"from convex_partition.c_interface import test_c_integration; test_c_integration()\"' to test integration"