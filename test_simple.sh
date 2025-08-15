#!/bin/bash

# DINOv2 Triton Server Test Script
# Simple curl-based tests for basic functionality

SERVER_URL="http://localhost:8000"
MODEL_NAME="dinov2_ensemble"

echo "🧪 DINOv2 Triton Server Test Script"
echo "=================================="

# Function to check if server is running
check_server() {
    echo "1️⃣ Checking server health..."
    if curl -s -f "$SERVER_URL/v2/health/ready" > /dev/null; then
        echo "✅ Server is healthy and ready"
        return 0
    else
        echo "❌ Server is not ready or not running"
        echo "   Make sure to start the server with: docker compose up triton-server"
        return 1
    fi
}

# Function to list models
list_models() {
    echo "2️⃣ Listing available models..."
    response=$(curl -s "$SERVER_URL/v2/models")
    if [ $? -eq 0 ]; then
        echo "✅ Available models:"
        echo "$response" | python3 -m json.tool
    else
        echo "❌ Failed to list models"
        return 1
    fi
}

# Function to check specific model
check_model() {
    echo "3️⃣ Checking model status..."
    response=$(curl -s "$SERVER_URL/v2/models/$MODEL_NAME")
    if [ $? -eq 0 ]; then
        echo "✅ Model '$MODEL_NAME' status:"
        echo "$response" | python3 -m json.tool
    else
        echo "❌ Failed to get model status"
        return 1
    fi
}

# Function to check server metadata
check_metadata() {
    echo "4️⃣ Checking server metadata..."
    response=$(curl -s "$SERVER_URL/v2")
    if [ $? -eq 0 ]; then
        echo "✅ Server metadata:"
        echo "$response" | python3 -m json.tool
    else
        echo "❌ Failed to get server metadata"
        return 1
    fi
}

# Function to test with a real image (if provided)
test_with_image() {
    local image_path="$1"
    
    if [ ! -f "$image_path" ]; then
        echo "❌ Image file not found: $image_path"
        return 1
    fi
    
    echo "5️⃣ Testing inference with image: $image_path"
    
    # Convert image to base64
    image_base64=$(base64 -i "$image_path")
    
    # Create JSON payload
    json_payload=$(cat <<EOF
{
    "inputs": [
        {
            "name": "IMAGE_BYTES",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["$image_base64"]
        }
    ]
}
EOF
)
    
    # Send request
    echo "🚀 Sending inference request..."
    start_time=$(date +%s.%N)
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$SERVER_URL/v2/models/$MODEL_NAME/infer")
    end_time=$(date +%s.%N)
    
    if [ $? -eq 0 ] && echo "$response" | grep -q "outputs"; then
        duration=$(echo "$end_time - $start_time" | bc)
        echo "✅ Inference successful!"
        echo "   Response time: ${duration}s"
        echo "   First few embedding values:"
        echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
embeddings = data['outputs'][0]['data']
print(f'   Embedding dimension: {len(embeddings)}')
print(f'   First 5 values: {embeddings[:5]}')
print(f'   Last 5 values: {embeddings[-5:]}')
"
    else
        echo "❌ Inference failed"
        echo "Response: $response"
        return 1
    fi
}

# Main execution
main() {
    check_server || exit 1
    echo ""
    
    list_models || exit 1
    echo ""
    
    check_model || exit 1
    echo ""
    
    check_metadata || exit 1
    echo ""
    
    # If image path provided as argument, test with it
    if [ $# -gt 0 ]; then
        test_with_image "$1"
    else
        echo "5️⃣ Skipping image inference test (no image provided)"
        echo "   To test with an image, run: $0 /path/to/your/image.jpg"
    fi
    
    echo ""
    echo "=================================="
    echo "✅ Basic tests completed!"
    echo ""
    echo "Next steps:"
    echo "- For comprehensive testing, run: python test_client.py"
    echo "- For testing with your own image: python test_client.py --image /path/to/image.jpg"
    echo "- For batch testing: python test_client.py --batch-test"
}

# Check if required tools are available
if ! command -v curl &> /dev/null; then
    echo "❌ curl is required but not installed"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 is required but not installed"
    exit 1
fi

# Run main function with all arguments
main "$@" 