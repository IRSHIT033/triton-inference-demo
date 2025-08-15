#!/bin/bash

echo "🔧 Setting up Triton Client Environment"
echo "======================================="

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not available. Please install Python and pip first."
    exit 1
fi

echo "📦 Installing Triton Client dependencies..."

# Install core Triton client dependencies
pip install tritonclient[all]>=2.40.0

# Install other required packages
pip install numpy>=1.21.0
pip install Pillow>=9.0.0
pip install requests>=2.25.0

# Optional but useful packages
echo "📦 Installing optional packages for enhanced features..."
pip install opencv-python>=4.5.0
pip install matplotlib>=3.3.0

echo ""
echo "✅ Triton Client setup completed!"
echo ""
echo "Available test clients:"
echo "1. Basic HTTP client:     python test_client.py"
echo "2. Advanced HTTP client:  python triton_client_advanced.py"
echo "3. Advanced gRPC client:  python triton_client_advanced.py --use-grpc"
echo "4. Shell-based tests:     ./test_simple.sh"
echo ""
echo "Example usage:"
echo "  python triton_client_advanced.py --image /path/to/image.jpg"
echo "  python triton_client_advanced.py --use-grpc --benchmark" 