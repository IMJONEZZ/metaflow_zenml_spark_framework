#!/usr/bin/env python3
"""
Hardware Monitor for AMD ROCm and DGX Spark System

This utility provides comprehensive monitoring of the system including:
- AMD ROCm GPU performance and compatibility detection
- NVIDIA CUDA GPU metrics (legacy support)
- CPU, memory, disk monitoring
- PyTorch device compatibility analysis

Enhanced with smart GPU detection for AMD ROCm hardware.

Usage:
    python hardware_monitor.py [--benchmark] [--duration SECONDS]
"""

import argparse
import json
import os
import platform
import time
from datetime import datetime
from typing import Any, Dict

import psutil

# Import smart device manager for AMD ROCm compatibility
import sys
sys.path.append('/home/imjonezz/Desktop/metaflow_zenml_spark_framework/src/utils')
from gpu_device_manager import get_device_with_fallback, GPUCompatibilityChecker

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not available
    class Fore:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"

    class Style:
        RESET_ALL = "\033[0m"
        BRIGHT = "\033[1m"


try:
    import gpustat

    GPUSTAT_AVAILABLE = True
except ImportError:
    print(Fore.YELLOW + "⚠️ gpustat not available, GPU monitoring will be limited")
    gpustat = None
    GPUSTAT_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    print(Fore.YELLOW + "⚠️ torch not available, PyTorch-specific features disabled")
    torch = None
    TORCH_AVAILABLE = False


class DGXSparkMonitor:
    def __init__(self):
        self.gpu_info = None
        if gpustat:
            try:
                self.gpu_info = gpustat.GPUStatCollection.new_query()
            except Exception as e:
                print(f"Could not query GPU stats: {e}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "platform": platform.uname()._asdict(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(
                    psutil.virtual_memory().available / (1024**3), 2
                ),
            },
        }

        # Add GPU information if available
        if self.gpu_info:
            info["gpu"] = []
            for gpu in self.gpu_info.gpus:
                try:
                    gpu_data = {
                        "name": gpu.name,
                        "memory_total_mb": getattr(gpu, "memory_total", None),
                        "memory_used_mb": getattr(gpu, "memory_used", None),
                        "temperature_celsius": getattr(gpu, "temperature", None),
                        "power_usage_watts": getattr(gpu, "power_draw", None),
                        "utilization_percent": getattr(gpu, "utilization", None),
                    }
                    info["gpu"].append(gpu_data)
                except Exception as e:
                    print(f"Warning: Could not extract GPU info for {gpu.name}: {e}")
                    continue

        # Add enhanced PyTorch device information (CUDA/ROCm)
        if TORCH_AVAILABLE and torch:
            info["pytorch_devices"] = {
                "cuda_available": torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
            }
            
            # Add ROCm-specific information
            if hasattr(torch.version, 'hip'):
                info["pytorch_devices"]["rocm_available"] = True
                info["pytorch_devices"]["hip_version"] = torch.version.hip if hasattr(torch.version, 'hip') else None
                info["pytorch_devices"]["rocm_arch"] = os.environ.get('PYTORCH_ROCM_ARCH', 'unknown')
            else:
                info["pytorch_devices"]["rocm_available"] = False
                
            # CUDA information (legacy support)
            if torch.cuda.is_available() and hasattr(torch, 'cuda'):
                info["pytorch_devices"]["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else None
                info["pytorch_devices"]["device_count"] = torch.cuda.device_count()
                info["pytorch_devices"]["current_device"] = torch.cuda.current_device() if hasattr(torch.cuda, 'current_device') else 0
                info["pytorch_devices"]["device_name"] = torch.cuda.get_device_name(0) if hasattr(torch.cuda, 'get_device_name') and torch.cuda.device_count() > 0 else None
                info["pytorch_devices"]["memory_allocated_gb"] = round(
                    torch.cuda.memory_allocated() / (1024**3), 2
                ) if hasattr(torch.cuda, 'memory_allocated') else None
                info["pytorch_devices"]["memory_reserved_gb"] = round(
                    torch.cuda.memory_reserved() / (1024**3), 2
                ) if hasattr(torch.cuda, 'memory_reserved') else None
            
            # Add smart device compatibility analysis
            try:
                temp_model = torch.nn.Sequential(
                    torch.nn.Linear(10, 5),  # Simple model for testing
                )
                
                device, compatibility_info = get_device_with_fallback(
                    model=temp_model,
                    force_device="auto",
                    batch_size=1
                )
                
                info["pytorch_devices"]["smart_device_analysis"] = compatibility_info.get('compatibility_analysis', {})
                info["pytorch_devices"]["recommended_device"] = str(device)
                
            except Exception as e:
                info["pytorch_devices"]["smart_analysis_error"] = str(e)

        return info

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "percent": memory.percent,
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
            },
            "disk": {
                "percent": (disk.used / disk.total) * 100,
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2),
            },
        }

        # Add GPU metrics if available
        if self.gpu_info:
            gpu_metrics = []
            for i, gpu in enumerate(self.gpu_info.gpus):
                gm = {
                    "gpu_id": i,
                    "name": gpu.name,
                }
                if hasattr(gpu, "utilization"):
                    gm["utilization_percent"] = gpu.utilization

                # Handle GPU memory stats with try-catch to avoid TypeError
                try:
                    memory_used = gpu.memory_used if gpu.memory_used is not None else 0
                    memory_total = getattr(gpu, "memory_total", 1)
                    if memory_total is None:
                        memory_total = 1

                    gm["memory_utilization"] = {
                        "used_mb": memory_used,
                        "total_mb": memory_total,
                        "percent": (memory_used / memory_total * 100)
                        if memory_total > 0
                        else 0,
                    }
                except (TypeError, AttributeError):
                    # Skip memory info if properties are not accessible
                    pass
                gpu_metrics.append(gm)
            metrics["gpu"] = gpu_metrics

        return metrics

    def run_basic_benchmark(self, duration: int = 60) -> Dict[str, Any]:
        """Run basic performance benchmarks."""
        print(f"Running {duration}-second hardware benchmark...")

        start_time = time.time()
        cpu_samples = []
        memory_samples = []

        # CPU benchmark
        start_cpu_time = time.time()
        cpu_result = sum(i * i for i in range(1000000))
        end_cpu_time = time.time()

        # Memory benchmark
        start_mem_time = time.time()
        data = [i for i in range(1000000)]
        memory_result = sum(data)
        end_mem_time = time.time()

        benchmark_results = {
            "duration_seconds": duration,
            "cpu_benchmark": {
                "computation_time_ms": round((end_cpu_time - start_cpu_time) * 1000, 2),
                "result": cpu_result,
            },
            "memory_benchmark": {
                "create_time_ms": round((end_mem_time - start_mem_time) * 1000, 2),
                "sum": memory_result,
            },
        }

        # Enhanced GPU benchmark with AMD ROCm support
        if TORCH_AVAILABLE and torch:
            try:
                # Use smart device manager for optimal GPU selection
                temp_model = torch.nn.Sequential(torch.nn.Linear(10, 5))
                device, _ = get_device_with_fallback(
                    model=temp_model,
                    force_device="auto",
                    batch_size=1
                )
                
                # GPU computation benchmark using optimal device
                gpu_start = time.time()
                
                if str(device) != "cpu":
                    x = torch.randn(1000, 1000).to(device)
                    y = torch.mm(x, x.t())
                    gpu_result = torch.sum(y).item()
                    gpu_end = time.time()
                    
                    benchmark_results["gpu_benchmark"] = {
                        "device": str(device),
                        "computation_time_ms": round((gpu_end - gpu_start) * 1000, 2),
                        "result": gpu_result,
                    }
                    
                    # Add ROCm-specific info if available
                    if hasattr(torch.version, 'hip'):
                        benchmark_results["gpu_benchmark"]["backend"] = "ROCm"
                        benchmark_results["gpu_benchmark"]["hip_version"] = getattr(torch.version, 'hip', None)
                    else:
                        benchmark_results["gpu_benchmark"]["backend"] = "CUDA"
                else:
                    # CPU fallback benchmark
                    x = torch.randn(500, 500)  # Smaller for CPU
                    y = torch.mm(x, x.t())
                    cpu_gpu_result = torch.sum(y).item()
                    
                    benchmark_results["cpu_fallback_benchmark"] = {
                        "reason": "GPU incompatible or unavailable",
                        "computation_time_ms": round((time.time() - gpu_start) * 1000, 2),
                        "result": cpu_gpu_result,
                    }
                    
            except Exception as e:
                benchmark_results["gpu_benchmark_error"] = str(e)

        return benchmark_results

    def monitor_continuous(self, duration: int = 30):
        """Monitor system continuously for specified duration."""
        print(f"Monitoring system for {duration} seconds... Press Ctrl+C to stop.")
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                metrics = self.get_current_metrics()
                print(
                    f"[{metrics['timestamp']}] CPU: {metrics['cpu_percent']:.1f}% | "
                    f"Memory: {metrics['memory']['percent']:.1f}%"
                )

                if "gpu" in metrics and metrics["gpu"]:
                    gpu = metrics["gpu"][0]
                    print(
                        f"GPU: {gpu.get('utilization_percent', 'N/A')}% | "
                        f"Memory: {gpu.get('memory_utilization', {}).get('percent', 'N/A'):.1f}%"
                    )

                time.sleep(2)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Hardware Monitor (AMD ROCm + NVIDIA CUDA)")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run hardware performance benchmarks"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--continuous",
        type=int,
        metavar="SECONDS",
        help="Monitor system continuously for specified seconds",
    )
    parser.add_argument(
        "--output", type=str, metavar="FILE", help="Output results to JSON file"
    )
    parser.add_argument(
        "--info", action="store_true", help="Show system information and exit"
    )

    args = parser.parse_args()

    monitor = DGXSparkMonitor()

    if args.info:
        info = monitor.get_system_info()
        print("\n=== System Hardware Information ===")
        
        # Enhanced system info display
        sys_info = info.get('system', {})
        platform_dict = sys_info.get('platform', {})
        
        # Handle different platform formats
        if isinstance(platform_dict, dict):
            system_name = platform_dict.get('system', 'Unknown')
            release = platform_dict.get('release', 'Unknown') 
        else:
            # Fallback for different formats
            platform_str = str(platform_dict)
            parts = platform_str.split(',')
            system_name = parts[0] if len(parts) > 0 else 'Unknown'
            release = parts[1].strip() if len(parts) > 1 else 'Unknown'
        
        print(f"Platform: {system_name} {release}")
        print(
            f"CPU Cores (Physical/Logical): {sys_info.get('cpu_count_physical', 'N/A')}/{sys_info.get('cpu_count_logical', 'N/A')}"
        )
        print(f"Memory: {sys_info.get('memory_total_gb', 'N/A')} GB total, {sys_info.get('memory_available_gb', 'N/A')} GB available")

        # Enhanced GPU information display
        if "gpu" in info and info["gpu"]:
            print("\n=== GPU Information ===")
            for i, gpu_data in enumerate(info["gpu"]):
                print(f"GPU {i}:")
                print(f"  Name: {gpu_data.get('name', 'Unknown')}")
                if gpu_data.get("memory_total_mb"):
                    print(
                        f"  Memory: {gpu_data.get('memory_used_mb', 'N/A')}/{gpu_data.get('memory_total_mb', 'N/A')} MB"
                    )
                if gpu_data.get("temperature_celsius"):
                    print(f"  Temperature: {gpu_data.get('temperature_celsius')}°C")
                if gpu_data.get("power_usage_watts"):
                    print(f"  Power: {gpu_data.get('power_usage_watts')}W")

        # Enhanced PyTorch device information
        if "pytorch_devices" in info:
            print("\n=== PyTorch Device Information ===")
            pt_info = info["pytorch_devices"]
            
            # ROCm information
            if pt_info.get('rocm_available', False):
                print("🦄 AMD ROCm Backend:")
                if pt_info.get('hip_version'):
                    print(f"  HIP Version: {pt_info['hip_version']}")
                if pt_info.get('rocm_arch'):
                    print(f"  Architecture: {pt_info['rocm_arch']}")
                    
            # CUDA information (legacy)
            if pt_info.get('cuda_available', False):
                print("🔥 NVIDIA CUDA Backend:")
                if pt_info.get('device_name'):
                    print(f"  Device: {pt_info['device_name']}")
                if pt_info.get('cuda_version'):
                    print(f"  CUDA Version: {pt_info['cuda_version']}")
                if pt_info.get('device_count'):
                    print(f"  Device Count: {pt_info['device_count']}")
                    
            # Smart device analysis
            if 'smart_device_analysis' in pt_info:
                print("🎯 Smart Device Analysis:")
                analysis = pt_info['smart_device_analysis']
                recommended_device = pt_info.get('recommended_device', 'unknown')
                
                print(f"  Recommended Device: {recommended_device}")
                if analysis.get('is_compatible'):
                    print("  ✅ Model compatibility: Good")
                else:
                    print("  ⚠️ Model compatibility: Issues detected")
                    
                if analysis.get('issues_found'):
                    print("  Issues found:")
                    for issue in analysis['issues_found']:
                        print(f"    - {issue.get('operation', 'Unknown')}: {issue.get('details', 'No details')}")

        return

    if args.benchmark:
        print("=== DGX Spark Hardware Benchmark ===")
        results = monitor.run_basic_benchmark(args.duration)

        print(
            f"CPU Computation: {results['cpu_benchmark']['computation_time_ms']:.2f} ms"
        )
        print(
            f"Memory Operation: {results['memory_benchmark']['create_time_ms']:.2f} ms"
        )

        if "gpu_benchmark" in results:
            print(
                f"GPU Computation: {results['gpu_benchmark']['computation_time_ms']:.2f} ms"
            )
            print(f"GPU Device: {results['gpu_benchmark']['device']}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.continuous:
        monitor.monitor_continuous(args.continuous)

    else:
        # Default: show current metrics
        metrics = monitor.get_current_metrics()
        print("=== Current System Metrics ===")
        print(f"CPU: {metrics['cpu_percent']:.1f}%")
        print(
            f"Memory: {metrics['memory']['used_gb']}/{metrics['memory']['available_gb']:.1f} GB "
            f"({metrics['memory']['percent']:.1f}% used)"
        )
        print(
            f"Disk: {metrics['disk']['free_gb']:.1f} GB free "
            f"({100 - metrics['disk']['percent']:.1f}% available)"
        )

        if "gpu" in metrics and metrics["gpu"]:
            for gpu in metrics["gpu"]:
                print(f"GPU: {gpu.get('name', 'Unknown')}")
                if "memory_utilization" in gpu:
                    mem = gpu["memory_utilization"]
                    print(
                        f"  Memory: {mem['used_mb']}/{mem['total_mb']} MB ({mem['percent']:.1f}% used)"
                    )
                util = gpu.get("utilization_percent", "N/A")
                print(f"  Utilization: {util}%")


if __name__ == "__main__":
    main()
