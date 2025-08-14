#!/usr/bin/env python3
"""
Script to convert DINO V2 ViT model to TensorRT format for Triton inference.
"""

import os
# Set environment variables before importing CUDA libraries
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TRT_BUILDER_OPTIMIZATION_LEVEL'] = '3'

import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
from transformers import Dinov2Model, Dinov2Config
import argparse
from pathlib import Path

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class DinoV2Wrapper(nn.Module):
    """Wrapper for DINO V2 model to extract features."""
    
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.eval()
    
    def forward(self, x):
        """Forward pass returning CLS token features."""
        outputs = self.model(x)
        # Extract CLS token features (first token)
        features = outputs.last_hidden_state[:, 0, :]
        return features

def build_tensorrt_engine(onnx_path, engine_path, max_batch_size=8, precision="fp16"):
    """Build TensorRT engine from ONNX model."""
    
    print("Initializing TensorRT engine builder...")
    
    # Force CUDA initialization through PyTorch first
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
        print("PyTorch CUDA initialized")
    
    try:
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        
        # Set memory pool size with reduced memory to avoid issues
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB instead of 1GB
        except AttributeError:
            # Fallback for older TensorRT versions
            config.max_workspace_size = 1 << 28  # 256MB
        
        # Set precision mode
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
        
        # Set optimization profile for dynamic batching
        profile = builder.create_optimization_profile()
        
        # Configure input dimensions (batch_size, channels, height, width)
        profile.set_shape("images", (1, 3, 224, 224), (4, 3, 224, 224), (max_batch_size, 3, 224, 224))
        config.add_optimization_profile(profile)
        
        # Build engine
        print("Building TensorRT engine... This may take a while.")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("ERROR: Failed to build TensorRT engine.")
            return None
        
        # Save engine
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to: {engine_path}")
        return engine
        
    except Exception as e:
        print(f"ERROR: TensorRT engine build failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_to_onnx(model, onnx_path, input_shape=(1, 3, 224, 224)):
    """Convert PyTorch model to ONNX format."""
    
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['features'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to: {onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert DINO V2 model to TensorRT")
    parser.add_argument("--model-name", default="facebook/dinov2-base", 
                       help="HuggingFace model name")
    parser.add_argument("--output-dir", default="./model_repository/dinov2_vit/1", 
                       help="Output directory for TensorRT engine")
    parser.add_argument("--max-batch-size", type=int, default=8, 
                       help="Maximum batch size")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16",
                       help="Precision mode")
    parser.add_argument("--onnx-only", action="store_true",
                       help="Only convert to ONNX, skip TensorRT conversion")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not args.onnx_only and not torch.cuda.is_available():
        print("WARNING: CUDA is not available. TensorRT conversion requires CUDA.")
        print("Consider using --onnx-only flag to generate ONNX model only.")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print(f"Loading DINO V2 model: {args.model_name}")
    model = DinoV2Wrapper(args.model_name)
    
    # Convert to ONNX
    onnx_path = output_dir / "model.onnx"
    print("Converting to ONNX...")
    convert_to_onnx(model, str(onnx_path))
    
    # Convert to TensorRT (if not ONNX-only mode)
    if not args.onnx_only:
        engine_path = output_dir / "model.plan"
        print("Converting to TensorRT...")
        engine = build_tensorrt_engine(
            str(onnx_path), 
            str(engine_path), 
            args.max_batch_size, 
            args.precision
        )
        
        if engine:
            print("Conversion completed successfully!")
            # Clean up ONNX file
            onnx_path.unlink()
        else:
            print("TensorRT conversion failed!")
            return 1
    else:
        print("ONNX conversion completed successfully!")
        print(f"ONNX model saved at: {onnx_path}")
        print("Skipping TensorRT conversion as requested.")
    
    return 0

if __name__ == "__main__":
    exit(main()) 