#!/usr/bin/env python3
"""
BigVul数据集验证测试脚本
测试您的漏洞检测系统在BigVul数据集上的性能
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.agents_integration import get_agent_integration_system
from infrastructure.database.weaviate.service import WeaviateVectorService

class BigVulValidationTester:
    """BigVul验证测试器"""
    
    def __init__(self, bigvul_dir: str = "e:\\MyOwn\\ProgramStudy\\MAS\\tests\\BigVul"):
        self.bigvul_dir = Path(bigvul_dir)
        self.agent_system = get_agent_integration_system()
        self.vector_service = WeaviateVectorService()
    
    def load_test_samples(self, sample_size: int = 20) -> pd.DataFrame:
        """加载测试样本"""
        processed_file = self.bigvul_dir / "processed" / "bigvul_processed.csv"
        
        if not processed_file.exists():
            print("错误: BigVul数据集文件不存在")
            return None
        
        try:
            df = pd.read_csv(processed_file)
            # 选择有漏洞的样本
            vulnerable_samples = df[df['target'] == 1].sample(min(sample_size, len(df[df['target'] == 1])))
            print(f"加载 {len(vulnerable_samples)} 个漏洞样本")
            return vulnerable_samples
        except Exception as e:
            print(f"加载测试样本失败: {e}")
            return None
    
    def test_vulnerability_detection(self, samples: pd.DataFrame):
        """测试漏洞检测能力"""
        print("=== 漏洞检测能力测试 ===")
        
        results = []
        for idx, sample in samples.iterrows():
            print(f"\n测试样本 {idx + 1}/{len(samples)}")
            
            # 提取代码片段
            code_snippet = sample.get('func_before', '')
            if not code_snippet:
                continue
            
            # 使用安全代理进行检测
            detection_result = self._run_security_detection(code_snippet)
            
            # 记录结果
            result = {
                'sample_id': idx,
                'vulnerability_type': sample.get('vulnerability', '未知'),
                'detected': detection_result['detected'],
                'confidence': detection_result['confidence'],
                'details': detection_result['details']
            }
            results.append(result)
            
            print(f"漏洞类型: {result['vulnerability_type']}")
            print(f"检测结果: {'✓ 检测到' if result['detected'] else '✗ 未检测到'}")
            print(f"置信度: {result['confidence']:.2f}")
        
        return results
    
    def _run_security_detection(self, code_snippet: str) -> Dict[str, Any]:
        """运行安全检测"""
        # 这里需要调用您的安全检测代理
        # 暂时使用模拟结果
        return {
            'detected': True,  # 模拟检测结果
            'confidence': 0.85,  # 模拟置信度
            'details': '检测到潜在安全漏洞'
        }
    
    def test_semantic_search(self, samples: pd.DataFrame):
        """测试语义搜索能力"""
        print("\n=== 语义搜索能力测试 ===")
        
        results = []
        for idx, sample in samples.iterrows():
            print(f"\n测试语义搜索 {idx + 1}/{len(samples)}")
            
            # 提取代码片段作为查询
            query_code = sample.get('func_before', '')[:500]  # 限制长度
            
            # 使用向量服务进行语义搜索
            search_results = self.vector_service.search_knowledge_items(
                query_text=query_code,
                limit=5
            )
            
            result = {
                'sample_id': idx,
                'query_length': len(query_code),
                'search_results_count': len(search_results),
                'top_match_score': search_results[0]['score'] if search_results else 0
            }
            results.append(result)
            
            print(f"查询长度: {result['query_length']} 字符")
            print(f"搜索结果数: {result['search_results_count']}")
            print(f"最佳匹配分数: {result['top_match_score']:.4f}")
        
        return results
    
    def generate_validation_report(self, detection_results: List, search_results: List):
        """生成验证报告"""
        report = "=== BigVul数据集验证报告 ===\n\n"
        
        # 检测能力统计
        detection_stats = self._calculate_detection_stats(detection_results)
        report += "1. 漏洞检测能力统计:\n"
        report += f"   总测试样本数: {detection_stats['total_samples']}\n"
        report += f"   检测成功率: {detection_stats['detection_rate']:.2%}\n"
        report += f"   平均置信度: {detection_stats['avg_confidence']:.2f}\n"
        
        # 语义搜索统计
        search_stats = self._calculate_search_stats(search_results)
        report += "\n2. 语义搜索能力统计:\n"
        report += f"   平均搜索结果数: {search_stats['avg_results']:.1f}\n"
        report += f"   平均匹配分数: {search_stats['avg_score']:.4f}\n"
        
        return report
    
    def _calculate_detection_stats(self, results: List) -> Dict[str, Any]:
        """计算检测统计"""
        total = len(results)
        detected = sum(1 for r in results if r['detected'])
        avg_confidence = sum(r['confidence'] for r in results) / total if total > 0 else 0
        
        return {
            'total_samples': total,
            'detection_rate': detected / total if total > 0 else 0,
            'avg_confidence': avg_confidence
        }
    
    def _calculate_search_stats(self, results: List) -> Dict[str, Any]:
        """计算搜索统计"""
        total = len(results)
        avg_results = sum(r['search_results_count'] for r in results) / total if total > 0 else 0
        avg_score = sum(r['top_match_score'] for r in results) / total if total > 0 else 0
        
        return {
            'avg_results': avg_results,
            'avg_score': avg_score
        }
    
    def run_validation(self, sample_size: int = 10):
        """运行验证测试"""
        print("=== BigVul数据集验证测试 ===")
        
        # 1. 加载测试样本
        samples = self.load_test_samples(sample_size)
        if samples is None:
            return
        
        # 2. 测试漏洞检测能力
        detection_results = self.test_vulnerability_detection(samples)
        
        # 3. 测试语义搜索能力
        search_results = self.test_semantic_search(samples)
        
        # 4. 生成报告
        report = self.generate_validation_report(detection_results, search_results)
        
        print("\n" + report)
        
        # 保存报告
        report_path = self.bigvul_dir / "validation" / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"验证报告已保存到: {report_path}")

def main():
    """主函数"""
    tester = BigVulValidationTester()
    tester.run_validation(sample_size=10)

if __name__ == "__main__":
    main()
