"""
Model Registry - Central registry for all AI models used in the system
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    CHAT = "chat"
    CODE = "code"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"

class ModelSize(Enum):
    SMALL = "small"      # < 1B parameters
    MEDIUM = "medium"    # 1B - 10B parameters
    LARGE = "large"      # > 10B parameters

@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    model_id: str
    model_type: ModelType
    size: ModelSize
    description: str
    min_transformers_version: str
    max_transformers_version: Optional[str] = None
    requires_trust_remote_code: bool = False
    gpu_memory_gb: float = 0.0
    special_requirements: List[str] = None
    known_issues: List[str] = None
    alternative_models: List[str] = None
    
    def __post_init__(self):
        if self.special_requirements is None:
            self.special_requirements = []
        if self.known_issues is None:
            self.known_issues = []
        if self.alternative_models is None:
            self.alternative_models = []

class ModelRegistry:
    """Registry for all models used in the system"""
    
    def __init__(self):
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize the model registry"""
        models = {}
        
        # Chat Models
        models["chatglm2-6b"] = ModelInfo(
            name="ChatGLM2-6B",
            model_id="THUDM/chatglm2-6b",
            model_type=ModelType.CHAT,
            size=ModelSize.MEDIUM,
            description="Chinese conversational AI model, 6B parameters",
            min_transformers_version="4.20.0",
            max_transformers_version="4.30.0",
            requires_trust_remote_code=True,
            gpu_memory_gb=12.0,
            special_requirements=["sentencepiece"],
            known_issues=[
                "KV cache incompatibility with transformers >= 4.40",
                "Tokenizer padding_side parameter issue",
                "get_masks method compatibility issues"
            ],
            alternative_models=["chatglm3-6b", "qwen2-7b"]
        )
        
        models["chatglm3-6b"] = ModelInfo(
            name="ChatGLM3-6B",
            model_id="THUDM/chatglm3-6b",
            model_type=ModelType.CHAT,
            size=ModelSize.MEDIUM,
            description="Updated Chinese conversational AI model with better compatibility",
            min_transformers_version="4.36.0",
            requires_trust_remote_code=True,
            gpu_memory_gb=12.0,
            special_requirements=["sentencepiece"],
            alternative_models=["qwen2-7b", "baichuan2-7b"]
        )
        
        models["qwen2-7b"] = ModelInfo(
            name="Qwen2-7B",
            model_id="Qwen/Qwen2-7B-Chat",
            model_type=ModelType.CHAT,
            size=ModelSize.MEDIUM,
            description="Alibaba's Qwen2 model with excellent Chinese and English capabilities",
            min_transformers_version="4.37.0",
            requires_trust_remote_code=True,
            gpu_memory_gb=14.0,
            alternative_models=["chatglm3-6b", "baichuan2-7b"]
        )
        
        models["baichuan2-7b"] = ModelInfo(
            name="Baichuan2-7B",
            model_id="baichuan-inc/Baichuan2-7B-Chat",
            model_type=ModelType.CHAT,
            size=ModelSize.MEDIUM,
            description="Baichuan's 7B chat model with strong Chinese capabilities",
            min_transformers_version="4.33.0",
            requires_trust_remote_code=True,
            gpu_memory_gb=14.0,
            alternative_models=["qwen2-7b", "chatglm3-6b"]
        )
        
        models["yi-6b-chat"] = ModelInfo(
            name="Yi-6B-Chat",
            model_id="01-ai/Yi-6B-Chat",
            model_type=ModelType.CHAT,
            size=ModelSize.MEDIUM,
            description="01.AI's Yi model with bilingual capabilities",
            min_transformers_version="4.34.0",
            requires_trust_remote_code=True,
            gpu_memory_gb=12.0,
            alternative_models=["qwen2-7b", "chatglm3-6b"]
        )
        
        # Code Models
        models["codebert"] = ModelInfo(
            name="CodeBERT",
            model_id="microsoft/codebert-base",
            model_type=ModelType.CODE,
            size=ModelSize.SMALL,
            description="Microsoft's CodeBERT for code understanding",
            min_transformers_version="4.0.0",
            gpu_memory_gb=2.0,
            alternative_models=["codet5-base"]
        )
        
        models["codet5-base"] = ModelInfo(
            name="CodeT5-Base",
            model_id="Salesforce/codet5-base",
            model_type=ModelType.CODE,
            size=ModelSize.SMALL,
            description="Salesforce's CodeT5 for code generation and understanding",
            min_transformers_version="4.0.0",
            gpu_memory_gb=2.0,
            alternative_models=["codebert"]
        )
        
        models["starcoder"] = ModelInfo(
            name="StarCoder",
            model_id="bigcode/starcoder",
            model_type=ModelType.CODE,
            size=ModelSize.LARGE,
            description="BigCode's StarCoder for code generation",
            min_transformers_version="4.28.0",
            requires_trust_remote_code=True,
            gpu_memory_gb=30.0,
            alternative_models=["codet5-base"]
        )
        
        # Embedding Models
        models["bge-large-zh"] = ModelInfo(
            name="BGE-Large-ZH",
            model_id="BAAI/bge-large-zh-v1.5",
            model_type=ModelType.EMBEDDING,
            size=ModelSize.MEDIUM,
            description="BAAI's Chinese text embedding model",
            min_transformers_version="4.0.0",
            gpu_memory_gb=4.0,
            alternative_models=["text2vec-large-chinese"]
        )
        
        models["text2vec-large-chinese"] = ModelInfo(
            name="Text2vec-Large-Chinese",
            model_id="shibing624/text2vec-large-chinese",
            model_type=ModelType.EMBEDDING,
            size=ModelSize.MEDIUM,
            description="Chinese text embedding model",
            min_transformers_version="4.0.0",
            gpu_memory_gb=3.0,
            alternative_models=["bge-large-zh"]
        )
        
        return models
    
    def get_model(self, model_key: str) -> Optional[ModelInfo]:
        """Get model info by key"""
        return self.models.get(model_key)
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """Get all models of a specific type"""
        return [model for model in self.models.values() if model.model_type == model_type]
    
    def get_chat_models(self) -> List[ModelInfo]:
        """Get all chat models"""
        return self.get_models_by_type(ModelType.CHAT)
    
    def get_code_models(self) -> List[ModelInfo]:
        """Get all code models"""
        return self.get_models_by_type(ModelType.CODE)
    
    def get_embedding_models(self) -> List[ModelInfo]:
        """Get all embedding models"""
        return self.get_models_by_type(ModelType.EMBEDDING)
    
    def get_compatible_models(self, transformers_version: str) -> List[ModelInfo]:
        """Get models compatible with a specific transformers version"""
        def version_to_number(version_str: str) -> int:
            try:
                parts = version_str.split('.')
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                return major * 100 + minor
            except (ValueError, IndexError):
                return 0
        
        current_version = version_to_number(transformers_version)
        compatible = []
        
        for model in self.models.values():
            min_version = version_to_number(model.min_transformers_version)
            max_version = version_to_number(model.max_transformers_version) if model.max_transformers_version else float('inf')
            
            if min_version <= current_version <= max_version:
                compatible.append(model)
        
        return compatible
    
    def get_problematic_models(self, transformers_version: str) -> List[ModelInfo]:
        """Get models that are problematic with current transformers version"""
        compatible = self.get_compatible_models(transformers_version)
        compatible_keys = [model.name for model in compatible]
        
        return [model for model in self.models.values() if model.name not in compatible_keys]
    
    def list_all_models(self) -> List[str]:
        """List all model keys"""
        return list(self.models.keys())
    
    def get_model_summary(self) -> Dict[str, Dict]:
        """Get summary of all models"""
        summary = {}
        for key, model in self.models.items():
            summary[key] = {
                "name": model.name,
                "type": model.model_type.value,
                "size": model.size.value,
                "gpu_memory_gb": model.gpu_memory_gb,
                "transformers_range": f"{model.min_transformers_version} - {model.max_transformers_version or 'latest'}",
                "known_issues_count": len(model.known_issues)
            }
        return summary

# Global registry instance
model_registry = ModelRegistry()

# Alternative Chat Models (Verified Compatible)
VERIFIED_ALTERNATIVES = {
    "chat_models": {
        "qwen1.5-7b-chat": {
            "model_id": "Qwen/Qwen1.5-7B-Chat",
            "type": "chat",
            "language": "multilingual",
            "size": "7B",
            "status": "verified",
            "recommended_as": "qwen2-replacement"
        },
        "baichuan2-7b-chat": {
            "model_id": "baichuan-inc/Baichuan2-7B-Chat", 
            "type": "chat",
            "language": "chinese",
            "size": "7B",
            "status": "verified"
        },
        "yi-6b-chat": {
            "model_id": "01-ai/Yi-6B-Chat",
            "type": "chat", 
            "language": "multilingual",
            "size": "6B",
            "status": "verified"
        }
    }
}
