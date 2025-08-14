#!/usr/bin/env python3
"""
Test script to check TensorRT CUDA initialization
"""

import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def test_tensorrt_cuda():
    print("=== CUDA and TensorRT Test ===")
    
    # Test PyTorch CUDA
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Test PyCUDA
    try:
        cuda.init()
        print(f"PyCUDA device count: {cuda.Device.count()}")
        device = cuda.Device(0)
        print(f"PyCUDA device name: {device.name()}")
        
        # Create CUDA context
        context = device.make_context()
        print("PyCUDA context created successfully")
        context.pop()
        print("PyCUDA context destroyed successfully")
    except Exception as e:
        print(f"PyCUDA error: {e}")
        return False
    
    # Test TensorRT
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        print("TensorRT Builder created successfully")
        
        # Try to create a simple network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        print("TensorRT Network created successfully")
        
        # Try to create builder config
        config = builder.create_builder_config()
        print("TensorRT Builder config created successfully")
        
        # Test memory allocation
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB
            print("TensorRT memory pool set successfully")
        except:
            config.max_workspace_size = 1 << 28  # Fallback
            print("TensorRT workspace size set successfully (fallback)")
        
        print("=== All TensorRT tests passed! ===")
        return True
        
    except Exception as e:
        print(f"TensorRT error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tensorrt_cuda()
    exit(0 if success else 1) 