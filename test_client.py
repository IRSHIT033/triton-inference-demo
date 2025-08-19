#!/usr/bin/env python3
"""
Simple test script for the Triton DinoV2 client
"""

import subprocess
import sys
import time

def test_client():
    """Test the client with error handling"""
    print("Testing DinoV2 Triton Client...")
    print("=" * 50)
    
    try:
        # Run the client
        result = subprocess.run([sys.executable, "client.py"], 
                              capture_output=True, 
                              text=True, 
                              timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Client executed successfully!")
        else:
            print(f"❌ Client failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("❌ Client timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error running client: {e}")

if __name__ == "__main__":
    test_client() 