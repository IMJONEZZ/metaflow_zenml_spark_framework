#!/usr/bin/env python3
"""
Hardware Monitor for DGX Spark System

This utility provides comprehensive monitoring of the NVIDIA DGX Spark system
including GPU performance, memory usage, temperature, and hardware specifications.

Usage:
    python hardware_monitor.py [--benchmark] [--duration SECONDS]
"""

import time
import argparse
import json
import psutil
import platform
from datetime import datetime
from typing import Dict, Any

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not available
    class Fore:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'

    class Style:
        RESET_ALL = '\033[0m'
        BRIGHT = '\033[1m'

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
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            }
        }

        # Add GPU information if available
        if self.gpu_info:
            info["gpu"] = []
            for gpu in self.gpu_info.gpus:
                try:
                    gpu_data = {
                        "name": gpu.name,
                        "memory_total_mb": getattr(gpu, 'memory_total', None),
                        "memory_used_mb": getattr(gpu, 'memory_used', None),
                        "temperature_celsius": getattr(gpu, 'temperature', None),
                        "power_usage_watts": getattr(gpu, 'power_draw', None),
                        "utilization_percent": getattr(gpu, 'utilization', None),
                    }
                    info["gpu"].append(gpu_data)
                except Exception as e:
                    print(f"Warning: Could not extract GPU info for {gpu.name}: {e}")
                    continue

        # Add PyTorch CUDA information if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info["pytorch_cuda"] = {
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                "memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
            }

        return info

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "percent": memory.percent,
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2)
            },
            "disk": {
                "percent": (disk.used / disk.total) * 100,
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2)
            }
        }

        # Add GPU metrics if available
        if self.gpu_info:
            gpu_metrics = []
            for i, gpu in enumerate(self.gpu_info.gpus):
                gm = {
                    "gpu_id": i,
                    "name": gpu.name,
                }
                if hasattr(gpu, 'utilization'):
                    gm["utilization_percent"] = gpu.utilization
                if hasattr(gpu, 'memory_used') and hasattr(gpu, 'memory_total'):
                    gm["memory_utilization"] = {
                        "used_mb": gpu.memory_used,
                        "total_mb": gpu.memory_total,
                        "percent": (gpu.memory_used / gpu.memory_total * 100) if gpu.memory_total > 0 else 0
                    }
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
        cpu_result = sum(i*i for i in range(1000000))
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
                "result": cpu_result
            },
            "memory_benchmark": {
                "create_time_ms": round((end_mem_time - start_mem_time) * 1000, 2),
                "sum": memory_result
            }
        }

        # GPU benchmark if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device = torch.device('cuda')
                # GPU computation benchmark
                gpu_start = time.time()
                x = torch.randn(1000, 1000).to(device)
                y = torch.mm(x, x.t())
                gpu_result = torch.sum(y).item()
                gpu_end = time.time()

                benchmark_results["gpu_benchmark"] = {
                    "device": torch.cuda.get_device_name(0),
                    "computation_time_ms": round((gpu_end - gpu_start) * 1000, 2),
                    "result": gpu_result
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
                print(f"[{metrics['timestamp']}] CPU: {metrics['cpu_percent']:.1f}% | "
                      f"Memory: {metrics['memory']['percent']:.1f}%")

                if 'gpu' in metrics and metrics['gpu']:
                    gpu = metrics['gpu'][0]
                    print(f"GPU: {gpu.get('utilization_percent', 'N/A')}% | "
                          f"Memory: {gpu.get('memory_utilization', {}).get('percent', 'N/A'):.1f}%")

                time.sleep(2)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")


def main():
    parser = argparse.ArgumentParser(description='DGX Spark Hardware Monitor')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run hardware performance benchmarks')
    parser.add_argument('--duration', type=int, default=60,
                       help='Benchmark duration in seconds (default: 60)')
    parser.add_argument('--continuous', type=int, metavar='SECONDS',
                       help='Monitor system continuously for specified seconds')
    parser.add_argument('--output', type=str, metavar='FILE',
                       help='Output results to JSON file')
    parser.add_argument('--info', action='store_true',
                       help='Show system information and exit')

    args = parser.parse_args()

    monitor = DGXSparkMonitor()

    if args.info:
        info = monitor.get_system_info()
        print("\n=== DGX Spark System Information ===")
        print(f"Platform: {info['system']['platform']}")
        print(f"CPU Cores (Physical/Logical): {info['system']['cpu_count_physical']}/{info['system']['cpu_count_logical']}")
        print(f"Memory: {info['system']['memory_total_gb']} GB")

        if 'gpu' in info:
            for i, gpu_data in enumerate(info['gpu']):
                print(f"GPU {i}:")
                print(f"  Name: {gpu_data['name']}")
                if gpu_data.get('memory_total_mb'):
                    print(f"  Memory: {gpu_data['memory_used_mb']}/{gpu_data['memory_total_mb']} MB")
                if gpu_data.get('temperature_celsius'):
                    print(f"  Temperature: {gpu_data['temperature_celsius']}°C")

        if 'pytorch_cuda' in info:
            pc = info['pytorch_cuda']
            print(f"PyTorch CUDA: {pc.get('device_name', 'Unknown')}")
            print(f"  CUDA Version: {pc.get('cuda_version', 'N/A')}")

        return

    if args.benchmark:
        print("=== DGX Spark Hardware Benchmark ===")
        results = monitor.run_basic_benchmark(args.duration)
        
        print(f"CPU Computation: {results['cpu_benchmark']['computation_time_ms']:.2f} ms")
        print(f"Memory Operation: {results['memory_benchmark']['create_time_ms']:.2f} ms")
        
        if 'gpu_benchmark' in results:
            print(f"GPU Computation: {results['gpu_benchmark']['computation_time_ms']:.2f} ms")
            print(f"GPU Device: {results['gpu_benchmark']['device']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.continuous:
        monitor.monitor_continuous(args.continuous)

    else:
        # Default: show current metrics
        metrics = monitor.get_current_metrics()
        print("=== Current System Metrics ===")
        print(f"CPU: {metrics['cpu_percent']:.1f}%")
        print(f"Memory: {metrics['memory']['used_gb']}/{metrics['memory']['available_gb']:.1f} GB "
              f"({metrics['memory']['percent']:.1f}% used)")
        print(f"Disk: {metrics['disk']['free_gb']:.1f} GB free "
              f"({100-metrics['disk']['percent']:.1f}% available)")

        if 'gpu' in metrics and metrics['gpu']:
            for gpu in metrics['gpu']:
                print(f"GPU: {gpu.get('name', 'Unknown')}")
                if 'memory_utilization' in gpu:
                    mem = gpu['memory_utilization']
                    print(f"  Memory: {mem['used_mb']}/{mem['total_mb']} MB ({mem['percent']:.1f}% used)")
                util = gpu.get('utilization_percent', 'N/A')
                print(f"  Utilization: {util}%")


if __name__ == "__main__":
    main()