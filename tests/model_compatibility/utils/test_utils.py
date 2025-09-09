"""
Compatibility Test Utilities
Common utilities for model compatibility testing
"""

import time
import logging
import traceback
import psutil
import torch
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    model_name: str
    status: str  # "pass", "fail", "skip", "warning"
    duration: float
    message: str
    details: Dict[str, Any] = None
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.details is None:
            self.details = {}

class MemoryMonitor:
    """Monitor memory usage during tests"""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.gpu_initial = None
        self.gpu_peak = None
    
    def start(self):
        """Start monitoring"""
        self.initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.gpu_initial = torch.cuda.memory_allocated() / (1024**3)  # GB
            self.gpu_peak = self.gpu_initial
    
    def update_peak(self):
        """Update peak memory usage"""
        current_memory = psutil.virtual_memory().used / (1024**3)
        if self.peak_memory is None or current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        if torch.cuda.is_available():
            current_gpu = torch.cuda.memory_allocated() / (1024**3)
            if current_gpu > self.gpu_peak:
                self.gpu_peak = current_gpu
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        self.update_peak()
        return {
            "initial_memory_gb": self.initial_memory or 0,
            "peak_memory_gb": self.peak_memory or 0,
            "memory_increase_gb": (self.peak_memory or 0) - (self.initial_memory or 0),
            "gpu_initial_gb": self.gpu_initial or 0,
            "gpu_peak_gb": self.gpu_peak or 0,
            "gpu_increase_gb": (self.gpu_peak or 0) - (self.gpu_initial or 0)
        }

@contextmanager
def test_timer():
    """Context manager for timing tests"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        return end_time - start_time

def safe_test_execution(test_func: Callable, test_name: str, model_name: str, 
                       timeout: int = 300) -> TestResult:
    """
    Safely execute a test function with error handling and monitoring
    """
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    start_time = time.time()
    
    try:
        # Execute the test
        result = test_func()
        duration = time.time() - start_time
        
        memory_stats = memory_monitor.get_stats()
        
        if result is True:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                status="pass",
                duration=duration,
                message="Test passed successfully",
                details={"memory_stats": memory_stats}
            )
        elif isinstance(result, dict):
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                status=result.get("status", "fail"),
                duration=duration,
                message=result.get("message", "Test completed"),
                details={**result.get("details", {}), "memory_stats": memory_stats},
                error=result.get("error")
            )
        else:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                status="fail",
                duration=duration,
                message="Test returned unexpected result",
                details={"memory_stats": memory_stats, "result": str(result)}
            )
    
    except Exception as e:
        duration = time.time() - start_time
        memory_stats = memory_monitor.get_stats()
        
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        return TestResult(
            test_name=test_name,
            model_name=model_name,
            status="fail",
            duration=duration,
            message=f"Test failed with exception: {error_msg}",
            details={"memory_stats": memory_stats, "traceback": error_traceback},
            error=error_msg
        )

def cleanup_model_resources():
    """Clean up model resources after testing"""
    try:
        # Clear Python garbage
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return True
    except Exception as e:
        logging.warning(f"Failed to cleanup resources: {e}")
        return False

def format_test_result(result: TestResult) -> str:
    """Format test result for display"""
    status_icons = {
        "pass": "✅",
        "fail": "❌", 
        "skip": "⏭️",
        "warning": "⚠️"
    }
    
    icon = status_icons.get(result.status, "❓")
    duration_str = f"{result.duration:.2f}s"
    
    output = f"{icon} {result.test_name} ({result.model_name}) - {duration_str}"
    
    if result.message:
        output += f"\n   💬 {result.message}"
    
    if result.details and "memory_stats" in result.details:
        stats = result.details["memory_stats"]
        if stats["memory_increase_gb"] > 0.1:  # Show if significant memory increase
            output += f"\n   📊 Memory: +{stats['memory_increase_gb']:.2f}GB"
        if stats["gpu_increase_gb"] > 0.1:
            output += f", GPU: +{stats['gpu_increase_gb']:.2f}GB"
    
    if result.error and result.status == "fail":
        output += f"\n   🚨 Error: {result.error}"
    
    return output

def create_compatibility_summary(results: List[TestResult]) -> Dict[str, Any]:
    """Create compatibility summary from test results"""
    total_tests = len(results)
    passed = len([r for r in results if r.status == "pass"])
    failed = len([r for r in results if r.status == "fail"])
    skipped = len([r for r in results if r.status == "skip"])
    warnings = len([r for r in results if r.status == "warning"])
    
    # Calculate total duration
    total_duration = sum(r.duration for r in results)
    
    # Get memory statistics
    max_memory_increase = 0
    max_gpu_increase = 0
    
    for result in results:
        if result.details and "memory_stats" in result.details:
            stats = result.details["memory_stats"]
            max_memory_increase = max(max_memory_increase, stats.get("memory_increase_gb", 0))
            max_gpu_increase = max(max_gpu_increase, stats.get("gpu_increase_gb", 0))
    
    # Group results by model
    by_model = {}
    for result in results:
        if result.model_name not in by_model:
            by_model[result.model_name] = {"pass": 0, "fail": 0, "skip": 0, "warning": 0}
        by_model[result.model_name][result.status] += 1
    
    return {
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "warnings": warnings,
        "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
        "total_duration": total_duration,
        "max_memory_increase_gb": max_memory_increase,
        "max_gpu_increase_gb": max_gpu_increase,
        "by_model": by_model,
        "timestamp": datetime.now().isoformat()
    }

def get_model_loading_config(model_id: str, force_cpu: bool = False) -> Dict[str, Any]:
    """Get optimized configuration for model loading"""
    config = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32 if force_cpu else torch.float16,
        "low_cpu_mem_usage": True,
    }
    
    if not force_cpu and torch.cuda.is_available():
        config["device_map"] = "auto"
    else:
        config["device_map"] = None
    
    # Model-specific configurations
    if "chatglm" in model_id.lower():
        config["use_cache"] = False  # Disable cache for compatibility
    
    if "qwen" in model_id.lower():
        config["use_flash_attention_2"] = False  # Disable flash attention for compatibility
    
    return config
