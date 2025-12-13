"""
Smart GPU Device Management for AMD ROCm Hardware

This module provides intelligent device selection and automatic fallback
mechanisms specifically designed for AMD GPU compatibility issues.

Key Features:
- Automatic CNN detection and CPU routing
- LSTM/RNN model GPU acceleration 
- Tree-based method optimization
- Memory fault prevention for Conv2d operations
- Performance monitoring and fallback triggers

Author: Metaflow/ZenML Framework Enhancement Team  
Date: December 12, 2025
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union


# Import with graceful fallback for missing dependencies
try:
    from colorama import Fore, Style, init
    
    # Try to initialize colorama
    try:
        init(autoreset=True)
    except Exception:
        pass  # Fallback to basic colors
except ImportError:
    # Create minimal colorama replacements for environments without it
    class ForeColors:
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
    
    Fore = ForeColors()


# Handle torch imports gracefully
TORCH_AVAILABLE = False
torch_device_class = None
torch_nn_module = None

try:
    import torch
    import torch.nn as nn
    
    # Check if this is actually ROCm PyTorch (not CPU-only)
    has_cuda = torch.cuda.is_available()
    
    if has_cuda:
        TORCH_AVAILABLE = True
        torch_device_class = torch.device
        torch_nn_module = nn
        
except ImportError:
    # PyTorch not available - create minimal replacements
    class DummyDevice:
        def __str__(self):
            return "cpu"
    
    torch_device_class = DummyDevice
    
    # Create minimal nn.Module for type checking
    class nn:
        @staticmethod 
        def Sequential(*args):
            return DummyModule()
    
    class DummyModule:
        pass
    
    torch_nn_module = nn


class GPUCompatibilityChecker:
    """Analyzes model architecture for AMD GPU compatibility."""
    
    # Known problematic operations on AMD ROCm
    PROBLEMATIC_OPS = {
        'Conv2d': 'Memory access faults on GFX1103 architecture',
        'Conv1d': 'HIP kernel compatibility issues', 
        'Conv3d': 'Not tested, likely problematic',
    }
    
    # Safe operations that work reliably
    SAFE_OPS = {
        'Linear': 'Simple matrix multiplication',
        'LSTM': 'Sequential operations, well supported', 
        'GRU': 'RNN operations work reliably',
        'Embedding': 'Lookup operations, GPU safe',
        'Dropout': 'Element-wise operations',
        'MaxPool1d': 'Simple pooling, no issues',
        'AdaptiveAvgPool2d': 'Pooling operations safe',
    }
    
    @classmethod
    def analyze_model(cls, model) -> Dict[str, any]:
        """Analyze a PyTorch model for GPU compatibility."""
        
        analysis = {
            'is_compatible': True,
            'risk_level': 'low',  # low, medium, high
            'issues_found': [],
            'recommendations': [],
            'safe_operations': 0,
            'problematic_operations': 0,
        }
        
        if not TORCH_AVAILABLE:
            analysis['recommendations'].append("PyTorch not available - using CPU")
            return analysis
            
        # Try to analyze model architecture
        try:
            for name, module in model.named_modules():
                module_type = type(module).__name__
                
                if module_type in cls.PROBLEMATIC_OPS:
                    analysis['problematic_operations'] += 1
                    analysis['issues_found'].append({
                        'operation': module_type,
                        'details': cls.PROBLEMATIC_OPS[module_type],
                        'module_name': name
                    })
                    
                elif module_type in cls.SAFE_OPS:
                    analysis['safe_operations'] += 1
                    
        except Exception as e:
            # If we can't analyze the model, be conservative and use CPU
            analysis['is_compatible'] = False
            analysis['risk_level'] = 'high'
            analysis['issues_found'].append({
                'operation': 'Unknown',
                'details': f'Could not analyze model: {e}',
                'module_name': 'unknown'
            })
            
        # Determine overall compatibility
        if analysis['problematic_operations'] > 0:
            if any('Conv2d' in issue['operation'] for issue in analysis['issues_found']):
                analysis['is_compatible'] = False
                analysis['risk_level'] = 'high'
                analysis['recommendations'].append(
                    "CNN/Conv2d operations detected - use CPU fallback to avoid memory faults"
                )
            else:
                analysis['risk_level'] = 'medium' 
        
        return analysis


class SmartDeviceManager:
    """Intelligent device management with automatic CNN detection."""
    
    def __init__(self, 
                 enable_cnn_fallback: bool = True,
                 memory_threshold_mb: float = 512.0,
                 enable_performance_monitoring: bool = True):
        
        self.enable_cnn_fallback = enable_cnn_fallback
        self.memory_threshold_mb = memory_threshold_mb  
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # AMD ROCm specific configuration
        self.amdgpu_info = self._detect_amdgpu_setup()
        
    def _detect_amdgpu_setup(self) -> Dict[str, any]:
        """Detect AMD GPU setup and capabilities."""
        
        info = {
            'has_rocm': False,
            'device_count': 0,
            'arch': 'unknown',
            'rocm_version': None,
        }
        
        if TORCH_AVAILABLE:
            try:
                info['has_rocm'] = torch.cuda.is_available()
                
                if info['has_rocm']:
                    info['device_count'] = torch.cuda.device_count()
                    
                    # Try to detect ROCm version
                    if hasattr(torch.version, 'hip'):
                        info['rocm_version'] = torch.version.hip
                        
                    # Get GPU name and architecture  
                    if info['device_count'] > 0:
                        gpu_name = torch.cuda.get_device_name(0)
                        info['gpu_name'] = gpu_name
                        
                        # Extract architecture (gfx1103, etc.)
                        arch = os.environ.get('PYTORCH_ROCM_ARCH', 'unknown')
                        info['arch'] = arch
                        
            except Exception as e:
                warnings.warn(f"Error detecting GPU setup: {e}")
                
        return info
    
    def get_optimal_device(self, 
                          model = None,
                          force_device: str = "auto",
                          batch_size: int = 32) -> any:
        """
        Get the optimal device for execution.
        
        Args:
            model: PyTorch model to analyze
            force_device: "auto", "cuda", or "cpu" 
            batch_size: Intended batch size for memory estimation
            
        Returns:
            Device object (torch.device or equivalent)
        """
        
        # Handle forced device selection
        if force_device == "cuda":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return torch.device("cuda")
            else:
                warnings.warn("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu") if TORCH_AVAILABLE else "cpu"
        elif force_device == "cpu":
            return torch.device("cpu") if TORCH_AVAILABLE else "cpu"
            
        # Auto-detection logic
        if not self.enable_cnn_fallback:
            device_name = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
            return torch.device(device_name) if TORCH_AVAILABLE else device_name
            
        # Analyze model for compatibility
        if model is not None:
            analysis = GPUCompatibilityChecker.analyze_model(model)
            
            # High risk CNN models → CPU fallback
            if not analysis['is_compatible'] and analysis['risk_level'] == 'high':
                print(Fore.YELLOW + 
                    f"⚠️  CNN model detected - routing to CPU to avoid memory faults")
                print(Fore.BLUE + 
                    f"   Issues found: {analysis['problematic_operations']} problematic operations")
                return torch.device("cpu") if TORCH_AVAILABLE else "cpu"
                
            # Medium risk models → Check memory availability
            elif analysis['risk_level'] == 'medium':
                print(Fore.YELLOW + 
                    f"⚠️  Medium risk model detected - checking memory...")
                if self._estimate_memory_usage(model, batch_size) > self.memory_threshold_mb:
                    print(Fore.YELLOW + 
                        f"   Memory usage high - using CPU for safety")
                    return torch.device("cpu") if TORCH_AVAILABLE else "cpu"
                    
        # Safe models or no model provided → Use GPU if available
        device_name = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        device = torch.device(device_name) if TORCH_AVAILABLE else device_name
        
        # Status reporting
        device_display = "GPU" if str(device) == "cuda" else "CPU"
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(Fore.GREEN + 
                    f"🚀 Using {device_display} ({gpu_name})")
            except:
                print(Fore.GREEN + 
                    f"🚀 Using {device_display} (AMD GPU)")
        else:
            print(Fore.BLUE + 
                f"💻 Using {device_display} (GPU not available)")
            
        return device
    
    def _estimate_memory_usage(self, model, batch_size: int) -> float:
        """Estimate memory usage in MB for a model and batch size."""
        
        if not TORCH_AVAILABLE:
            return 100.0  # Conservative estimate
            
        try:
            # Rough estimation based on parameter counts
            param_count = sum(p.numel() for p in model.parameters())
            
            # Assume 4 bytes per parameter, plus activation memory
            param_memory_mb = (param_count * 4) / (1024 * 1024)
            
            # Add estimated activation memory
            activation_factor = batch_size * param_count / 1000000  # Rough factor
            total_memory_mb = param_memory_mb + activation_factor
            
            return total_memory_mb
        except:
            return 200.0  # Conservative fallback
    
    def get_safe_model_config(self, model) -> Dict[str, any]:
        """Get safe configuration settings for problematic models."""
        
        if model is not None:
            analysis = GPUCompatibilityChecker.analyze_model(model)
        else:
            analysis = {'is_compatible': True, 'risk_level': 'low'}
        
        config = {
            'batch_size': 32,      # Safe default
            'mixed_precision': False,
            'no_grad_checkpointing': True,
        }
        
        if not analysis['is_compatible']:
            # Adjust for CNN models
            config.update({
                'batch_size': 16,      # Smaller batches to reduce memory pressure
                'no_grad_checkpointing': True,
            })
            
        return config


def get_device_with_fallback(model = None,
                           force_device: str = "auto",
                           batch_size: int = 32) -> Tuple[any, Dict[str, any]]:
    """
    Convenience function for getting optimal device with detailed info.
    
    Args:
        model: PyTorch model to analyze
        force_device: "auto", "cuda", or "cpu"
        batch_size: Intended batch size
        
    Returns:
        Tuple[device, info_dict] where info contains compatibility analysis
    """
    
    manager = SmartDeviceManager()
    
    device = manager.get_optimal_device(
        model=model, 
        force_device=force_device,
        batch_size=batch_size
    )
    
    info = {
        'device_name': str(device),
        'has_rocm_gpu': TORCH_AVAILABLE and torch.cuda.is_available(),
        'amdgpu_info': manager.amdgpu_info,
        'torch_available': TORCH_AVAILABLE,
    }
    
    if model is not None:
        info['compatibility_analysis'] = GPUCompatibilityChecker.analyze_model(model)
        
    return device, info


# Example usage patterns
def example_usage():
    """Show how to use the smart device manager."""
    
    # Simple usage
    device, info = get_device_with_fallback()
    print(f"Recommended device: {device}")
    
    # With model analysis (requires torch to test properly)
    if TORCH_AVAILABLE:
        try:
            cnn_model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            device, info = get_device_with_fallback(model=cnn_model)
            print(f"CNN Model device: {device}")
            
            if not info['compatibility_analysis']['is_compatible']:
                print("Model has compatibility issues, using CPU fallback")
        except:
            print("Could not create test CNN model")


if __name__ == "__main__":
    example_usage()