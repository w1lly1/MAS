#!/usr/bin/env python3
"""
Comprehensive Model Compatibility Test Suite
Runs compatibility tests for all models in the registry
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.model_compatibility.model_registry import model_registry
from tests.model_compatibility.utils.test_utils import (
    create_compatibility_summary, format_test_result
)

# Import individual test modules
from tests.model_compatibility.individual_model_tests.test_chatglm3 import run_chatglm3_tests
from tests.model_compatibility.individual_model_tests.test_qwen2 import run_qwen2_tests
from tests.model_compatibility.individual_model_tests.test_codebert import run_codebert_tests

class CompatibilityTestSuite:
    """Main compatibility test suite"""
    
    def __init__(self):
        self.all_results = []
        self.test_modules = {
            "chatglm3-6b": run_chatglm3_tests,
            "qwen2-7b": run_qwen2_tests,
            "codebert": run_codebert_tests,
            # Add more test modules as they are created
        }
    
    def run_environment_check(self):
        """Run environment compatibility check"""
        print("🔍 Running Environment Check...")
        try:
            from tests.model_compatibility.environment_check import generate_environment_report, print_report
            
            report = generate_environment_report()
            print_report(report)
            
            # Save environment report
            os.makedirs("tests/model_compatibility/results", exist_ok=True)
            with open("tests/model_compatibility/results/environment_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            return report
            
        except Exception as e:
            print(f"❌ Environment check failed: {e}")
            return None
    
    def run_model_tests(self, models_to_test: List[str] = None):
        """Run tests for specified models or all available models"""
        
        if models_to_test is None:
            models_to_test = list(self.test_modules.keys())
        
        print(f"\n🚀 Starting Model Compatibility Tests")
        print(f"📋 Models to test: {', '.join(models_to_test)}")
        print("=" * 80)
        
        for model_key in models_to_test:
            if model_key not in self.test_modules:
                print(f"⚠️ No test module found for {model_key}, skipping...")
                continue
            
            print(f"\n{'='*20} Testing {model_key.upper()} {'='*20}")
            
            try:
                # Run the test module
                test_func = self.test_modules[model_key]
                model_results = test_func()
                
                # Add to overall results
                self.all_results.extend(model_results)
                
                # Print model summary
                model_passed = len([r for r in model_results if r.status == "pass"])
                model_total = len(model_results)
                print(f"\n📊 {model_key} Summary: {model_passed}/{model_total} tests passed")
                
            except Exception as e:
                print(f"❌ Failed to run tests for {model_key}: {e}")
                import traceback
                print(f"🐛 Traceback: {traceback.format_exc()}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive compatibility report"""
        
        if not self.all_results:
            return {"error": "No test results available"}
        
        # Create basic summary
        summary = create_compatibility_summary(self.all_results)
        
        # Add detailed analysis
        model_analysis = {}
        for result in self.all_results:
            model_name = result.model_name
            if model_name not in model_analysis:
                model_analysis[model_name] = {
                    "tests": [],
                    "status": "unknown",
                    "recommendation": ""
                }
            
            model_analysis[model_name]["tests"].append({
                "test_name": result.test_name,
                "status": result.status,
                "duration": result.duration,
                "message": result.message
            })
        
        # Determine overall status and recommendations for each model
        for model_name, analysis in model_analysis.items():
            test_results = analysis["tests"]
            passed_tests = [t for t in test_results if t["status"] == "pass"]
            failed_tests = [t for t in test_results if t["status"] == "fail"]
            
            total_tests = len(test_results)
            passed_count = len(passed_tests)
            
            if passed_count == total_tests:
                analysis["status"] = "fully_compatible"
                analysis["recommendation"] = "Ready for production use"
            elif passed_count >= total_tests * 0.8:
                analysis["status"] = "mostly_compatible"
                analysis["recommendation"] = "Suitable with minor limitations"
            elif passed_count >= total_tests * 0.5:
                analysis["status"] = "partially_compatible"
                analysis["recommendation"] = "Requires fixes or workarounds"
            else:
                analysis["status"] = "incompatible"
                analysis["recommendation"] = "Not recommended for current environment"
        
        # Add transformers version info
        try:
            import transformers
            transformers_info = {
                "version": transformers.__version__,
                "compatibility_notes": self._get_transformers_compatibility_notes()
            }
        except ImportError:
            transformers_info = {"error": "transformers not installed"}
        
        return {
            "summary": summary,
            "model_analysis": model_analysis,
            "transformers_info": transformers_info,
            "recommendations": self._generate_recommendations(model_analysis),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_transformers_compatibility_notes(self) -> List[str]:
        """Get transformers version specific compatibility notes"""
        try:
            import transformers
            version = transformers.__version__
            major, minor = map(int, version.split('.')[:2])
            version_number = major * 100 + minor
            
            notes = []
            
            if version_number >= 456:
                notes.append("Very recent transformers version - some models may have compatibility issues")
            elif version_number >= 440:
                notes.append("Post-4.40 version - KV cache system has been restructured")
                notes.append("ChatGLM2-6B likely incompatible due to cache changes")
            elif version_number >= 436:
                notes.append("Good compatibility with modern models like ChatGLM3, Qwen2")
            elif version_number >= 430:
                notes.append("Decent compatibility but may miss some newer model features")
            else:
                notes.append("Older transformers version - newer models may not be supported")
            
            return notes
            
        except Exception:
            return ["Could not analyze transformers version"]
    
    def _generate_recommendations(self, model_analysis: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on test results"""
        recommendations = []
        
        # Count compatible models
        fully_compatible = len([m for m in model_analysis.values() if m["status"] == "fully_compatible"])
        mostly_compatible = len([m for m in model_analysis.values() if m["status"] == "mostly_compatible"])
        total_models = len(model_analysis)
        
        if fully_compatible == total_models:
            recommendations.append("🎉 All models are fully compatible - proceed with current setup")
        elif fully_compatible + mostly_compatible >= total_models * 0.8:
            recommendations.append("✅ Most models are compatible - minor adjustments may be needed")
        else:
            recommendations.append("⚠️ Multiple compatibility issues found - consider environment changes")
        
        # Specific recommendations
        incompatible_models = [name for name, analysis in model_analysis.items() 
                             if analysis["status"] == "incompatible"]
        
        if incompatible_models:
            recommendations.append(f"🔧 Consider alternatives for incompatible models: {', '.join(incompatible_models)}")
        
        # Transformers version recommendations
        try:
            import transformers
            version = transformers.__version__
            major, minor = map(int, version.split('.')[:2])
            version_number = major * 100 + minor
            
            if version_number >= 456:
                recommendations.append("📦 Consider downgrading transformers to 4.40-4.50 range for better compatibility")
            elif version_number < 436:
                recommendations.append("📦 Consider upgrading transformers to 4.36+ for modern model support")
                
        except Exception:
            pass
        
        return recommendations
    
    def save_results(self):
        """Save test results to files"""
        os.makedirs("tests/model_compatibility/results", exist_ok=True)
        
        # Save detailed results
        detailed_results = [
            {
                "test_name": r.test_name,
                "model_name": r.model_name,
                "status": r.status,
                "duration": r.duration,
                "message": r.message,
                "details": r.details,
                "error": r.error,
                "timestamp": r.timestamp
            }
            for r in self.all_results
        ]
        
        with open("tests/model_compatibility/results/detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        with open("tests/model_compatibility/results/compatibility_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved:")
        print(f"  📄 Detailed results: tests/model_compatibility/results/detailed_results.json")
        print(f"  📊 Compatibility report: tests/model_compatibility/results/compatibility_report.json")
    
    def print_final_summary(self):
        """Print final test summary"""
        if not self.all_results:
            print("❌ No test results to summarize")
            return
        
        report = self.generate_comprehensive_report()
        
        print(f"\n" + "="*80)
        print(f"🎯 FINAL COMPATIBILITY SUMMARY")
        print(f"="*80)
        
        # Overall statistics
        summary = report["summary"]
        print(f"📊 Overall Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']} ({summary['success_rate']:.1f}%)")
        print(f"  Failed: {summary['failed']}")
        print(f"  Warnings: {summary['warnings']}")
        print(f"  Duration: {summary['total_duration']:.1f}s")
        
        # Model-by-model analysis
        print(f"\n📋 Model Analysis:")
        for model_name, analysis in report["model_analysis"].items():
            status_icons = {
                "fully_compatible": "✅",
                "mostly_compatible": "🟢", 
                "partially_compatible": "🟡",
                "incompatible": "❌"
            }
            icon = status_icons.get(analysis["status"], "❓")
            print(f"  {icon} {model_name}: {analysis['recommendation']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
        
        print(f"\n🔗 Next Steps:")
        print(f"  1. Review detailed results in tests/model_compatibility/results/")
        print(f"  2. Address compatibility issues for critical models")
        print(f"  3. Consider environment adjustments if needed")
        print(f"  4. Run production deployment tests for compatible models")

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description="Model Compatibility Test Suite")
    parser.add_argument("--models", nargs="*", help="Specific models to test (default: all)")
    parser.add_argument("--skip-env", action="store_true", help="Skip environment check")
    parser.add_argument("--save-only", action="store_true", help="Only save results, don't print summary")
    
    args = parser.parse_args()
    
    suite = CompatibilityTestSuite()
    
    # Run environment check unless skipped
    if not args.skip_env:
        suite.run_environment_check()
    
    # Run model tests
    suite.run_model_tests(args.models)
    
    # Save results
    suite.save_results()
    
    # Print summary unless suppressed
    if not args.save_only:
        suite.print_final_summary()

if __name__ == "__main__":
    main()
