#!/usr/bin/env python3
"""
Intel NPU Access Program for Core Ultra 7 155U
This program demonstrates how to access and use the NPU on Intel Core Ultra processors.
"""

import sys
import time
import numpy as np
from pathlib import Path

try:
    import openvino as ov
    from openvino import Core, Type
    print("OpenVINO imported successfully!")
except ImportError:
    print("OpenVINO not installed. Please install with: pip install openvino")
    sys.exit(1)

class NPUManager:
    def __init__(self):
        self.core = Core()
        self.npu_device = None
        self.available_devices = []
        
    def check_npu_availability(self):
        """Check if NPU is available on the system"""
        print("Checking available devices...")
        self.available_devices = self.core.available_devices
        
        print(f"Available devices: {self.available_devices}")
        
        # Look for NPU device
        for device in self.available_devices:
            if "NPU" in device:
                self.npu_device = device
                print(f"✅ NPU found: {device}")
                return True
        
        print("❌ NPU not found. Available devices:", self.available_devices)
        return False
    
    def get_device_info(self):
        """Get detailed information about available devices"""
        print("\n=== Device Information ===")
        for device in self.available_devices:
            try:
                device_name = self.core.get_property(device, "FULL_DEVICE_NAME")
                print(f"Device: {device}")
                print(f"  Full name: {device_name}")
                
                if "NPU" in device:
                    # Get NPU-specific properties
                    try:
                        optimization_capabilities = self.core.get_property(device, "OPTIMIZATION_CAPABILITIES")
                        print(f"  Optimization capabilities: {optimization_capabilities}")
                    except:
                        pass
                print()
            except Exception as e:
                print(f"  Could not get info for {device}: {e}")
    
    def create_simple_model(self):
        """Create a simple test model for NPU"""
        print("\n=== Creating Simple Test Model ===")
        
        # Create a simple addition model
        from openvino.runtime import opset8 as ops
        
        # Input parameters
        input_shape = [1, 10]
        
        # Create input
        input_tensor = ops.parameter(input_shape, Type.f32, name="input")
        
        # Create a simple operation (add constant)
        constant = ops.constant(np.ones(input_shape, dtype=np.float32), Type.f32)
        add_op = ops.add(input_tensor, constant, name="add")
        
        # Create result
        result = ops.result(add_op, name="output")
        
        # Create model
        model = ov.Model([result], [input_tensor], "simple_add_model")
        
        print("✅ Simple model created successfully")
        return model
    
    def run_npu_inference(self, model):
        """Run inference on NPU"""
        if not self.npu_device:
            print("❌ No NPU device available")
            return False
        
        print(f"\n=== Running Inference on {self.npu_device} ===")
        
        try:
            # Compile model for NPU
            print("Compiling model for NPU...")
            compiled_model = self.core.compile_model(model, self.npu_device)
            
            # Create inference request
            infer_request = compiled_model.create_infer_request()
            
            # Prepare input data
            input_data = np.random.rand(1, 10).astype(np.float32)
            print(f"Input data: {input_data}")
            
            # Run inference
            print("Running inference...")
            start_time = time.time()
            infer_request.infer({0: input_data})
            inference_time = time.time() - start_time
            
            # Get results
            output = infer_request.get_output_tensor(0).data
            print(f"Output data: {output}")
            print(f"Inference time: {inference_time:.4f} seconds")
            
            print("✅ NPU inference completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error during NPU inference: {e}")
            return False
    
    def benchmark_devices(self, model):
        """Benchmark the model on different devices"""
        print("\n=== Device Benchmark ===")
        
        test_devices = ["CPU"]
        if self.npu_device:
            test_devices.append(self.npu_device)
        
        # Add GPU if available
        if any("GPU" in device for device in self.available_devices):
            test_devices.append("GPU")
        
        input_data = np.random.rand(1, 10).astype(np.float32)
        num_runs = 10
        
        results = {}
        
        for device in test_devices:
            if device not in self.available_devices and device != "CPU":
                continue
                
            try:
                print(f"\nTesting {device}...")
                compiled_model = self.core.compile_model(model, device)
                infer_request = compiled_model.create_infer_request()
                
                # Warmup
                for _ in range(3):
                    infer_request.infer({0: input_data})
                
                # Benchmark
                times = []
                for _ in range(num_runs):
                    start = time.time()
                    infer_request.infer({0: input_data})
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                results[device] = avg_time
                print(f"{device}: {avg_time:.4f}s average ({num_runs} runs)")
                
            except Exception as e:
                print(f"❌ Error testing {device}: {e}")
        
        # Print comparison
        if results:
            print(f"\n=== Benchmark Results ===")
            sorted_results = sorted(results.items(), key=lambda x: x[1])
            for device, time_ms in sorted_results:
                print(f"{device}: {time_ms:.4f}s")

def main():
    print("Intel NPU Access Program")
    print("========================")
    
    # Initialize NPU manager
    npu_manager = NPUManager()
    
    # Check NPU availability
    if not npu_manager.check_npu_availability():
        print("\nNote: If you expect NPU to be available, make sure:")
        print("1. You have the latest Intel graphics drivers installed")
        print("2. OpenVINO is properly installed with NPU support")
        print("3. Your system BIOS has NPU enabled")
    
    # Get device information
    npu_manager.get_device_info()
    
    # Create a simple test model
    model = npu_manager.create_simple_model()
    
    # Run NPU inference if available
    if npu_manager.npu_device:
        npu_manager.run_npu_inference(model)
    else:
        print("\n⚠️  NPU not available, running on CPU instead...")
        try:
            compiled_model = npu_manager.core.compile_model(model, "CPU")
            infer_request = compiled_model.create_infer_request()
            input_data = np.random.rand(1, 10).astype(np.float32)
            infer_request.infer({0: input_data})
            output = infer_request.get_output_tensor(0).data
            print(f"CPU inference successful. Output: {output}")
        except Exception as e:
            print(f"❌ CPU inference failed: {e}")
    
    # Benchmark different devices
    npu_manager.benchmark_devices(model)
    
    print("\n=== Next Steps ===")
    print("1. Try loading pre-trained models (ONNX, OpenVINO IR)")
    print("2. Experiment with different model types (vision, NLP, etc.)")
    print("3. Use Intel's Model Zoo for optimized models")
    print("4. Check Intel's NPU documentation for advanced features")

if __name__ == "__main__":
    main()