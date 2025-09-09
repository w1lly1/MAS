#!/usr/bin/env python3
"""
Individual Model Test: ChatGLM3-6B
Tests ChatGLM3-6B compatibility with current transformers version
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from tests.model_compatibility.utils.test_utils import (
    safe_test_execution, cleanup_model_resources, 
    format_test_result, get_model_loading_config
)
from tests.model_compatibility.model_registry import model_registry

class ChatGLM3Tester:
    """ChatGLM3-6B compatibility tester"""
    
    def __init__(self):
        self.model_info = model_registry.get_model("chatglm3-6b")
        self.model = None
        self.tokenizer = None
        self.config = None
    
    def test_config_loading(self):
        """Test configuration loading"""
        try:
            self.config = AutoConfig.from_pretrained(
                self.model_info.model_id,
                trust_remote_code=True
            )
            
            # Check essential attributes
            required_attrs = ['num_layers', 'hidden_size', 'num_attention_heads']
            missing_attrs = [attr for attr in required_attrs if not hasattr(self.config, attr)]
            
            if missing_attrs:
                return {
                    "status": "fail",
                    "message": f"Missing config attributes: {missing_attrs}",
                    "details": {"config_type": str(type(self.config))}
                }
            
            return {
                "status": "pass",
                "message": "Configuration loaded successfully",
                "details": {
                    "config_type": str(type(self.config)),
                    "num_layers": getattr(self.config, 'num_layers', 'N/A'),
                    "hidden_size": getattr(self.config, 'hidden_size', 'N/A'),
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Config loading failed: {str(e)}",
                "error": str(e)
            }
    
    def test_tokenizer_loading(self):
        """Test tokenizer loading and basic functionality"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_info.model_id,
                trust_remote_code=True
            )
            
            # Test basic encoding/decoding
            test_text = "你好，我是ChatGLM3助手"
            tokens = self.tokenizer.encode(test_text)
            decoded = self.tokenizer.decode(tokens)
            
            # Test special tokens
            special_tokens = {
                "bos_token": getattr(self.tokenizer, 'bos_token', None),
                "eos_token": getattr(self.tokenizer, 'eos_token', None),
                "pad_token": getattr(self.tokenizer, 'pad_token', None),
            }
            
            return {
                "status": "pass",
                "message": "Tokenizer loaded and functioning",
                "details": {
                    "tokenizer_type": str(type(self.tokenizer)),
                    "vocab_size": len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 'unknown',
                    "test_text": test_text,
                    "decoded_text": decoded,
                    "special_tokens": special_tokens
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Tokenizer test failed: {str(e)}",
                "error": str(e)
            }
    
    def test_model_loading(self):
        """Test model loading"""
        try:
            loading_config = get_model_loading_config(self.model_info.model_id)
            
            self.model = AutoModel.from_pretrained(
                self.model_info.model_id,
                **loading_config
            )
            
            # Test model properties
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "status": "pass",
                "message": "Model loaded successfully",
                "details": {
                    "model_type": str(type(self.model)),
                    "parameter_count": param_count,
                    "trainable_parameters": trainable_params,
                    "device": str(next(self.model.parameters()).device),
                    "dtype": str(next(self.model.parameters()).dtype)
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Model loading failed: {str(e)}",
                "error": str(e)
            }
    
    def test_basic_inference(self):
        """Test basic inference capability"""
        if not self.model or not self.tokenizer:
            return {
                "status": "skip",
                "message": "Model or tokenizer not loaded, skipping inference test"
            }
        
        try:
            # Test basic chat functionality
            test_query = "你好"
            
            # Use the model's chat method if available
            if hasattr(self.model, 'chat'):
                response, history = self.model.chat(
                    self.tokenizer,
                    test_query,
                    history=[],
                    max_length=100,
                    temperature=0.8
                )
                
                return {
                    "status": "pass",
                    "message": "Basic inference successful",
                    "details": {
                        "query": test_query,
                        "response": response[:100] + "..." if len(response) > 100 else response,
                        "response_length": len(response),
                        "method": "chat"
                    }
                }
            else:
                # Fallback to generate method
                inputs = self.tokenizer.encode(test_query, return_tensors="pt")
                if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                
                return {
                    "status": "pass",
                    "message": "Basic inference successful",
                    "details": {
                        "query": test_query,
                        "response": response,
                        "method": "generate"
                    }
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Inference test failed: {str(e)}",
                "error": str(e)
            }
    
    def test_batch_processing(self):
        """Test batch processing capability"""
        if not self.model or not self.tokenizer:
            return {
                "status": "skip",
                "message": "Model or tokenizer not loaded, skipping batch test"
            }
        
        try:
            # Test batch tokenization
            test_queries = ["你好", "今天天气怎么样？", "请介绍一下人工智能"]
            
            # Batch encode
            encoded_batch = self.tokenizer(
                test_queries,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            
            # Test forward pass
            with torch.no_grad():
                outputs = self.model(**encoded_batch)
            
            return {
                "status": "pass",
                "message": "Batch processing successful",
                "details": {
                    "batch_size": len(test_queries),
                    "input_shape": list(encoded_batch['input_ids'].shape),
                    "output_shape": list(outputs.last_hidden_state.shape) if hasattr(outputs, 'last_hidden_state') else "unknown"
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Batch processing failed: {str(e)}",
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up resources"""
        self.model = None
        self.tokenizer = None
        self.config = None
        cleanup_model_resources()

def run_chatglm3_tests():
    """Run all ChatGLM3-6B tests"""
    print("🚀 Starting ChatGLM3-6B Compatibility Tests")
    print("=" * 60)
    
    tester = ChatGLM3Tester()
    results = []
    
    # Test sequence
    tests = [
        ("Config Loading", tester.test_config_loading),
        ("Tokenizer Loading", tester.test_tokenizer_loading),
        ("Model Loading", tester.test_model_loading),
        ("Basic Inference", tester.test_basic_inference),
        ("Batch Processing", tester.test_batch_processing),
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        
        result = safe_test_execution(
            test_func=test_func,
            test_name=test_name,
            model_name="ChatGLM3-6B"
        )
        
        results.append(result)
        print(format_test_result(result))
        
        # Stop on critical failures
        if result.status == "fail" and test_name in ["Config Loading", "Tokenizer Loading"]:
            print(f"❌ Critical test failed, stopping test sequence")
            break
    
    # Cleanup
    tester.cleanup()
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    passed = len([r for r in results if r.status == "pass"])
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! ChatGLM3-6B is compatible.")
    elif passed > 0:
        print("⚠️ Some tests passed. Check individual results.")
    else:
        print("❌ All tests failed. ChatGLM3-6B is not compatible.")
    
    return results

if __name__ == "__main__":
    run_chatglm3_tests()
