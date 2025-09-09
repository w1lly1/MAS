#!/usr/bin/env python3
"""
Individual Model Test: CodeBERT
Tests CodeBERT compatibility for code analysis tasks
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

class CodeBERTTester:
    """CodeBERT compatibility tester"""
    
    def __init__(self):
        self.model_info = model_registry.get_model("codebert")
        self.model = None
        self.tokenizer = None
        self.config = None
    
    def test_config_loading(self):
        """Test CodeBERT configuration loading"""
        try:
            self.config = AutoConfig.from_pretrained(self.model_info.model_id)
            
            # Check BERT-style attributes
            bert_attrs = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size']
            missing_attrs = [attr for attr in bert_attrs if not hasattr(self.config, attr)]
            
            if missing_attrs:
                return {
                    "status": "fail",
                    "message": f"Missing BERT config attributes: {missing_attrs}",
                    "details": {"config_type": str(type(self.config))}
                }
            
            return {
                "status": "pass",
                "message": "CodeBERT configuration loaded successfully",
                "details": {
                    "config_type": str(type(self.config)),
                    "num_hidden_layers": self.config.num_hidden_layers,
                    "hidden_size": self.config.hidden_size,
                    "vocab_size": self.config.vocab_size,
                    "model_type": getattr(self.config, 'model_type', 'roberta')
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"CodeBERT config loading failed: {str(e)}",
                "error": str(e)
            }
    
    def test_tokenizer_loading(self):
        """Test CodeBERT tokenizer loading and code tokenization"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_info.model_id)
            
            # Test different types of code
            code_samples = [
                "def hello_world():\n    print('Hello, World!')",
                "function calculateSum(a, b) { return a + b; }",
                "public class HelloWorld { public static void main(String[] args) { System.out.println('Hello'); } }",
                "import numpy as np\narray = np.array([1, 2, 3])"
            ]
            
            tokenization_results = {}
            for i, code in enumerate(code_samples):
                tokens = self.tokenizer.encode(code)
                decoded = self.tokenizer.decode(tokens)
                tokenization_results[f"code_sample_{i}"] = {
                    "original": code,
                    "token_count": len(tokens),
                    "decoded_matches": decoded.strip() == code.strip()
                }
            
            # Test special tokens
            special_tokens = {
                "bos_token": getattr(self.tokenizer, 'bos_token', None),
                "eos_token": getattr(self.tokenizer, 'eos_token', None),
                "sep_token": getattr(self.tokenizer, 'sep_token', None),
                "cls_token": getattr(self.tokenizer, 'cls_token', None),
                "pad_token": getattr(self.tokenizer, 'pad_token', None),
                "mask_token": getattr(self.tokenizer, 'mask_token', None),
            }
            
            return {
                "status": "pass",
                "message": "CodeBERT tokenizer loaded and functioning for code",
                "details": {
                    "tokenizer_type": str(type(self.tokenizer)),
                    "vocab_size": len(self.tokenizer),
                    "tokenization_results": tokenization_results,
                    "special_tokens": special_tokens
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"CodeBERT tokenizer test failed: {str(e)}",
                "error": str(e)
            }
    
    def test_model_loading(self):
        """Test CodeBERT model loading"""
        try:
            loading_config = get_model_loading_config(self.model_info.model_id)
            
            self.model = AutoModel.from_pretrained(
                self.model_info.model_id,
                **loading_config
            )
            
            # Analyze model structure
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Check for BERT-style components
            has_embeddings = hasattr(self.model, 'embeddings')
            has_encoder = hasattr(self.model, 'encoder')
            has_pooler = hasattr(self.model, 'pooler')
            
            return {
                "status": "pass",
                "message": "CodeBERT model loaded successfully",
                "details": {
                    "model_type": str(type(self.model)),
                    "parameter_count": param_count,
                    "parameter_count_millions": round(param_count / 1e6, 1),
                    "trainable_parameters": trainable_params,
                    "device": str(next(self.model.parameters()).device),
                    "dtype": str(next(self.model.parameters()).dtype),
                    "has_embeddings": has_embeddings,
                    "has_encoder": has_encoder,
                    "has_pooler": has_pooler
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"CodeBERT model loading failed: {str(e)}",
                "error": str(e)
            }
    
    def test_code_embedding(self):
        """Test CodeBERT code embedding capability"""
        if not self.model or not self.tokenizer:
            return {
                "status": "skip",
                "message": "Model or tokenizer not loaded, skipping embedding test"
            }
        
        try:
            # Test code embedding for different programming languages
            code_snippets = [
                "def process_data(data): return data.strip().lower()",
                "function processData(data) { return data.trim().toLowerCase(); }",
                "public String processData(String data) { return data.trim().toLowerCase(); }"
            ]
            
            embedding_results = {}
            
            for i, code in enumerate(code_snippets):
                # Tokenize
                inputs = self.tokenizer(
                    code,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Use CLS token embedding or mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                    # Mean pooling
                    embeddings = hidden_states.mean(dim=1)
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs[0].mean(dim=1)
                
                embedding_results[f"code_{i}"] = {
                    "code": code[:50] + "..." if len(code) > 50 else code,
                    "embedding_shape": list(embeddings.shape),
                    "embedding_norm": float(torch.norm(embeddings)),
                    "input_length": inputs['input_ids'].shape[1]
                }
            
            return {
                "status": "pass",
                "message": "CodeBERT embedding generation successful",
                "details": {
                    "embedding_results": embedding_results,
                    "total_codes": len(code_snippets)
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"CodeBERT embedding test failed: {str(e)}",
                "error": str(e)
            }
    
    def test_code_similarity(self):
        """Test CodeBERT code similarity detection"""
        if not self.model or not self.tokenizer:
            return {
                "status": "skip",
                "message": "Model or tokenizer not loaded, skipping similarity test"
            }
        
        try:
            # Test semantically similar code in different languages
            code_pairs = [
                (
                    "def add_numbers(a, b): return a + b",
                    "function addNumbers(a, b) { return a + b; }"
                ),
                (
                    "for i in range(10): print(i)",
                    "for (int i = 0; i < 10; i++) { System.out.println(i); }"
                )
            ]
            
            similarity_results = {}
            
            for i, (code1, code2) in enumerate(code_pairs):
                # Get embeddings for both codes
                def get_embedding(code):
                    inputs = self.tokenizer(
                        code,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    
                    if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        return outputs.last_hidden_state.mean(dim=1)
                
                emb1 = get_embedding(code1)
                emb2 = get_embedding(code2)
                
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
                
                similarity_results[f"pair_{i}"] = {
                    "code1": code1,
                    "code2": code2,
                    "similarity": float(similarity),
                    "similar": float(similarity) > 0.7
                }
            
            return {
                "status": "pass",
                "message": "CodeBERT similarity detection working",
                "details": {
                    "similarity_results": similarity_results,
                    "total_pairs": len(code_pairs)
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"CodeBERT similarity test failed: {str(e)}",
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up resources"""
        self.model = None
        self.tokenizer = None
        self.config = None
        cleanup_model_resources()

def run_codebert_tests():
    """Run all CodeBERT tests"""
    print("🚀 Starting CodeBERT Compatibility Tests")
    print("=" * 60)
    
    tester = CodeBERTTester()
    results = []
    
    # Test sequence
    tests = [
        ("Config Loading", tester.test_config_loading),
        ("Tokenizer Loading", tester.test_tokenizer_loading),
        ("Model Loading", tester.test_model_loading),
        ("Code Embedding", tester.test_code_embedding),
        ("Code Similarity", tester.test_code_similarity),
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        
        result = safe_test_execution(
            test_func=test_func,
            test_name=test_name,
            model_name="CodeBERT"
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
        print("🎉 All tests passed! CodeBERT is compatible.")
    elif passed > 0:
        print("⚠️ Some tests passed. CodeBERT is partially compatible.")
    else:
        print("❌ All tests failed. CodeBERT is not compatible.")
    
    return results

if __name__ == "__main__":
    run_codebert_tests()
