"""
Device management utilities for GPU acceleration on Apple Silicon and other platforms
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .logging import get_logger


@dataclass
class DeviceInfo:
    """Information about the current device configuration"""
    device_type: str  # 'mps', 'cuda', 'cpu'
    device_name: str
    memory_gb: Optional[float]
    is_available: bool
    pytorch_version: Optional[str]


class DeviceManager:
    """
    Manages device selection and provides clear reporting of GPU/CPU usage
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.device_info = self._detect_device()
        self._log_device_info()
    
    def _detect_device(self) -> DeviceInfo:
        """Detect the best available device"""
        
        if not TORCH_AVAILABLE:
            return DeviceInfo(
                device_type="cpu",
                device_name="CPU (PyTorch not available)",
                memory_gb=None,
                is_available=True,
                pytorch_version=None
            )
        
        pytorch_version = torch.__version__
        
        # Check for Apple Metal Performance Shaders (MPS)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                # Test MPS functionality
                test_tensor = torch.tensor([1.0]).to('mps')
                del test_tensor
                
                return DeviceInfo(
                    device_type="mps",
                    device_name="Apple Metal Performance Shaders (M1/M2/M3)",
                    memory_gb=self._get_unified_memory_gb(),
                    is_available=True,
                    pytorch_version=pytorch_version
                )
            except Exception as e:
                self.logger.warning(f"MPS detected but failed test: {e}")
        
        # Check for CUDA
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return DeviceInfo(
                device_type="cuda",
                device_name=f"NVIDIA {device_name}",
                memory_gb=memory_gb,
                is_available=True,
                pytorch_version=pytorch_version
            )
        
        # Fallback to CPU
        return DeviceInfo(
            device_type="cpu",
            device_name="CPU (No GPU acceleration available)",
            memory_gb=None,
            is_available=True,
            pytorch_version=pytorch_version
        )
    
    def _get_unified_memory_gb(self) -> Optional[float]:
        """Estimate unified memory on Apple Silicon"""
        try:
            import psutil
            # On Apple Silicon, GPU shares unified memory with CPU
            total_memory = psutil.virtual_memory().total / (1024**3)
            return total_memory
        except ImportError:
            return None
    
    def _log_device_info(self):
        """Log detailed device information"""
        info = self.device_info
        
        self.logger.info("🖥️  Device Configuration:")
        self.logger.info(f"   Device: {info.device_name}")
        self.logger.info(f"   Type: {info.device_type.upper()}")
        
        if info.memory_gb:
            self.logger.info(f"   Memory: {info.memory_gb:.1f} GB")
        
        if info.pytorch_version:
            self.logger.info(f"   PyTorch: {info.pytorch_version}")
        
        # Log specific capabilities
        if info.device_type == "mps":
            self.logger.info("   Capabilities: GPU-accelerated neural networks, embeddings")
        elif info.device_type == "cuda":
            self.logger.info("   Capabilities: Full GPU acceleration")
        else:
            self.logger.info("   Capabilities: CPU-only processing")
    
    def get_torch_device(self) -> str:
        """Get the PyTorch device string"""
        return self.device_info.device_type if self.device_info.device_type != "cpu" else "cpu"
    
    def get_sentence_transformers_device(self) -> Optional[str]:
        """Get device string for sentence-transformers"""
        if self.device_info.device_type in ["mps", "cuda"]:
            return self.device_info.device_type
        return None  # Let sentence-transformers auto-detect
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.device_info.device_type in ["mps", "cuda"]
    
    def log_library_device_usage(self, library_name: str, device_used: str, note: str = ""):
        """Log which device a specific library is using"""
        emoji = "🚀" if device_used.upper() != "CPU" else "💻"
        note_str = f" ({note})" if note else ""
        self.logger.info(f"   {emoji} {library_name}: {device_used.upper()}{note_str}")
    
    def get_device_summary(self) -> Dict[str, Any]:
        """Get a summary of device information"""
        return {
            "device_type": self.device_info.device_type,
            "device_name": self.device_info.device_name,
            "memory_gb": self.device_info.memory_gb,
            "pytorch_available": TORCH_AVAILABLE,
            "pytorch_version": self.device_info.pytorch_version,
            "gpu_available": self.is_gpu_available()
        }


# Global device manager instance
_device_manager = None

def get_device_manager() -> DeviceManager:
    """Get the global device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager