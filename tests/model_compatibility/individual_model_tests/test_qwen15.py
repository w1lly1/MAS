#!/usr/bin/env python3
"""
Individual Model Test: Qwen1.5-7B
Tests Qwen1.5-7B compatibility with current transformers version
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from tests.model_compatibility.utils.test_utils import (
    safe_test_execution, cleanup_model_resources, 
    format_test_result, get_model_loading_config
)
from tests.model_compatibility.model_registry import model_registry

class Qwen15Tester:
    """Qwen1.5-7B compatibility tester"""
    
    def __init__(self):
        self.model_info = model_registry.get_model("qwen1.5-7b")
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
            
            # Check Qwen1.5 specific attributes
            qwen_attrs = ['hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads']
            missing_attrs = [attr for attr in qwen_attrs if not hasattr(self.config, attr)]
            
            if missing_attrs:
                return {
                    "status": "fail",
                    "message": f"Missing Qwen1.5 config attributes: {missing_attrs}",
                    "details": {"config_type": str(type(self.config))}
                }
            
            return {
                "status": "pass",
                "message": "Qwen1.5 configuration loaded successfully",
                "details": {
                    "config_type": str(type(self.config)),
                    "num_hidden_layers": self.config.num_hidden_layers,
                    "hidden_size": self.config.hidden_size,
                    "vocab_size": getattr(self.config, 'vocab_size', 'N/A'),
                    "model_type": getattr(self.config, 'model_type', 'N/A')
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Qwen1.5 config loading failed: {str(e)}",
                "error": str(e)
            }
    
    def test_tokenizer_loading(self):
        """Test tokenizer loading and Qwen1.5-specific functionality"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_info.model_id,
                trust_remote_code=True
            )
            
            # Test Chinese and English text
            test_texts = [
                "你好，我是Qwen1.5助手",
                "Hello, I am Qwen1.5 assistant",
                "代码分析是重要的软件质量保证手段"
            ]
            
            tokenizer_results = {}
            for i, text in enumerate(test_texts):
                tokens = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(tokens)
                tokenizer_results[f"test_{i}"] = {
                    "original": text,
                    "token_count": len(tokens),
                    "decoded": decoded
                }
            
            # Test special tokens and chat format
            special_tokens = {
                "bos_token": getattr(self.tokenizer, 'bos_token', None),
                "eos_token": getattr(self.tokenizer, 'eos_token', None),
                "pad_token": getattr(self.tokenizer, 'pad_token', None),
                "unk_token": getattr(self.tokenizer, 'unk_token', None),
            }
            
            return {
                "status": "pass",
                "message": "Qwen1.5 tokenizer loaded and functioning",
                "details": {
                    "tokenizer_type": str(type(self.tokenizer)),
                    "vocab_size": len(self.tokenizer),
                    "test_results": tokenizer_results,
                    "special_tokens": special_tokens
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Qwen1.5 tokenizer test failed: {str(e)}",
                "error": str(e)
            }
    
    def test_model_loading(self):
        """Test Qwen1.5 model loading"""
        try:
            loading_config = get_model_loading_config(self.model_info.model_id)
            
            # Qwen1.5 specific configurations (minimal to avoid compatibility issues)
            loading_config.update({
                "use_cache": True,
                "output_attentions": False,
                "output_hidden_states": False,
            })
            
            # Remove any problematic parameters
            loading_config.pop('use_flash_attention_2', None)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_info.model_id,
                **loading_config
            )
            
            # Analyze model structure
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Check for Qwen1.5 specific components
            has_transformer = hasattr(self.model, 'transformer') or hasattr(self.model, 'model')
            has_lm_head = hasattr(self.model, 'lm_head')
            
            return {
                "status": "pass",
                "message": "Qwen1.5 model loaded successfully",
                "details": {
                    "model_type": str(type(self.model)),
                    "parameter_count": param_count,
                    "parameter_count_millions": round(param_count / 1e6, 1),
                    "trainable_parameters": trainable_params,
                    "device": str(next(self.model.parameters()).device),
                    "dtype": str(next(self.model.parameters()).dtype),
                    "has_transformer": has_transformer,
                    "has_lm_head": has_lm_head
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Qwen1.5 model loading failed: {str(e)}",
                "error": str(e)
            }
    
    def test_text_generation(self):
        """Test Qwen1.5 text generation capability"""
        if not self.model or not self.tokenizer:
            return {
                "status": "skip",
                "message": "Model or tokenizer not loaded, skipping generation test"
            }
        
        try:
            # Test different types of prompts
            test_prompts = [
                "你好，请介绍一下自己",
                "What is artificial intelligence?",
                "请解释什么是代码质量"
            ]
            
            generation_results = {}
            
            for i, prompt in enumerate(test_prompts):
                # Encode prompt
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.8,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.shape[1]:], 
                    skip_special_tokens=True
                )
                
                generation_results[f"prompt_{i}"] = {
                    "prompt": prompt,
                    "generated": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                    "generated_length": len(generated_text)
                }
            
            return {
                "status": "pass",
                "message": "Qwen1.5 text generation successful",
                "details": {
                    "generation_results": generation_results,
                    "total_prompts": len(test_prompts)
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Qwen1.5 generation test failed: {str(e)}",
                "error": str(e)
            }
    
    def test_chat_format(self):
        """Test Qwen1.5 chat format functionality"""
        if not self.tokenizer:
            return {
                "status": "skip",
                "message": "Tokenizer not loaded, skipping chat format test"
            }
        
        try:
            # Test chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "user", "content": "你好，请介绍一下代码分析的重要性"}
                ]
                
                chat_formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                return {
                    "status": "pass",
                    "message": "Qwen1.5 chat format working",
                    "details": {
                        "original_messages": messages,
                        "formatted_chat": chat_formatted[:200] + "..." if len(chat_formatted) > 200 else chat_formatted,
                        "method": "apply_chat_template"
                    }
                }
            else:
                # Manual chat format
                user_message = "你好，请介绍一下代码分析的重要性"
                manual_format = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
                
                return {
                    "status": "warning",
                    "message": "Chat template not available, using manual format",
                    "details": {
                        "manual_format": manual_format,
                        "method": "manual"
                    }
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Chat format test failed: {str(e)}",
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up resources"""
        self.model = None
        self.tokenizer = None
        self.config = None
        cleanup_model_resources()

def run_qwen15_tests():
    """Run all Qwen1.5-7B tests"""
    print("🚀 Starting Qwen1.5-7B Compatibility Tests")
    print("=" * 60)
    
    tester = Qwen15Tester()
    results = []
    
    # Test sequence
    tests = [
        ("Config Loading", tester.test_config_loading),
        ("Tokenizer Loading", tester.test_tokenizer_loading),
        ("Model Loading", tester.test_model_loading),
        ("Text Generation", tester.test_text_generation),
        ("Chat Format", tester.test_chat_format),
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        
        result = safe_test_execution(
            test_func=test_func,
            test_name=test_name,
            model_name="Qwen1.5-7B"
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
    warnings = len([r for r in results if r.status == "warning"])
    total = len(results)
    print(f"Passed: {passed}/{total}")
    if warnings > 0:
        print(f"Warnings: {warnings}")
    
    if passed == total:
        print("🎉 All tests passed! Qwen1.5-7B is fully compatible.")
    elif passed + warnings == total:
        print("✅ Tests passed with warnings. Qwen1.5-7B is mostly compatible.")
    elif passed > 0:
        print("⚠️ Some tests passed. Check individual results.")
    else:
        print("❌ All tests failed. Qwen1.5-7B is not compatible.")
    
    return results

if __name__ == "__main__":
    run_qwen15_tests()
