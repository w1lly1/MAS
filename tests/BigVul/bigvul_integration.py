#!/usr/bin/env python3
"""
BigVul数据集与MAS系统集成脚本
将BigVul漏洞数据导入到您的IssuePattern知识库中
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.weaviate.service import WeaviateVectorService
from infrastructure.database.vector_sync import IssuePatternSyncService

class BigVulIntegration:
    """BigVul数据集与MAS系统集成器"""
    
    def __init__(self, bigvul_dir: str = "e:\\MyOwn\\ProgramStudy\\MAS\\tests\\BigVul"):
        self.bigvul_dir = Path(bigvul_dir)
        self.db_service = DatabaseService()
        self.vector_service = WeaviateVectorService()
        self.sync_service = IssuePatternSyncService(
            self.db_service, self.vector_service
        )
    
    def load_bigvul_data(self) -> pd.DataFrame:
        """加载BigVul数据"""
        processed_file = self.bigvul_dir / "processed" / "bigvul_processed.csv"
        
        if not processed_file.exists():
            print("错误: BigVul数据集文件不存在，请先运行下载脚本")
            return None
        
        try:
            df = pd.read_csv(processed_file)
            print(f"BigVul数据集加载成功: {df.shape}")
            return df
        except Exception as e:
            print(f"加载BigVul数据集失败: {e}")
            return None
    
    def convert_to_issue_patterns(self, df: pd.DataFrame, sample_size: int = 100) -> List[Dict[str, Any]]:
        """将BigVul数据转换为IssuePattern格式"""
        print(f"转换BigVul数据为IssuePattern格式 (样本大小: {sample_size})")
        
        # 随机抽样
        sample_df = df.sample(min(sample_size, len(df)))
        
        issue_patterns = []
        for idx, row in sample_df.iterrows():
            # 根据BigVul字段映射到IssuePattern
            issue_pattern = {
                "title": f"BigVul漏洞模式-{idx}",
                "error_type": self._map_vulnerability_type(row.get('vulnerability', '')),
                "severity": self._map_severity(row),
                "language": self._detect_language(row),
                "framework": "",
                "error_description": self._build_error_description(row),
                "problematic_pattern": row.get('func_before', ''),
                "solution": self._build_solution(row),
                "file_pattern": "",
                "class_pattern": "",
                "tags": f"BigVul,{row.get('vulnerability', '')}",
                "status": "active"
            }
            issue_patterns.append(issue_pattern)
        
        print(f"成功转换 {len(issue_patterns)} 个IssuePattern")
        return issue_patterns
    
    def _map_vulnerability_type(self, vul_type: str) -> str:
        """映射漏洞类型"""
        vul_mapping = {
            'CWE-119': 'buffer_error',
            'CWE-125': 'out_of_bounds_read',
            'CWE-787': 'out_of_bounds_write',
            'CWE-20': 'input_validation',
            'CWE-79': 'xss',
            'CWE-89': 'sql_injection',
            'CWE-94': 'code_injection',
            'CWE-400': 'resource_exhaustion',
            'CWE-476': 'null_pointer',
            'CWE-502': 'deserialization'
        }
        return vul_mapping.get(vul_type, 'security_vulnerability')
    
    def _map_severity(self, row) -> str:
        """映射严重程度"""
        # 根据漏洞类型和上下文确定严重程度
        vul_type = row.get('vulnerability', '')
        if vul_type in ['CWE-119', 'CWE-125', 'CWE-787']:
            return 'high'
        elif vul_type in ['CWE-89', 'CWE-94']:
            return 'critical'
        else:
            return 'medium'
    
    def _detect_language(self, row) -> str:
        """检测编程语言"""
        # 根据代码片段特征检测语言
        code = row.get('func_before', '')
        if 'def ' in code and 'import ' in code:
            return 'python'
        elif 'function ' in code and 'var ' in code:
            return 'javascript'
        elif 'public ' in code and 'class ' in code:
            return 'java'
        else:
            return 'unknown'
    
    def _build_error_description(self, row) -> str:
        """构建错误描述"""
        vul_type = row.get('vulnerability', '未知漏洞')
        return f"BigVul数据集中的{vul_type}类型漏洞。问题代码: {row.get('func_before', '')[:200]}..."
    
    def _build_solution(self, row) -> str:
        """构建解决方案"""
        return f"修复建议: 检查代码中的安全漏洞，使用安全的编程实践。修复后代码: {row.get('func_after', '')[:200]}..."
    
    def import_to_database(self, issue_patterns: List[Dict[str, Any]]):
        """将IssuePattern导入数据库"""
        print("开始导入IssuePattern到数据库...")
        
        success_count = 0
        for pattern in issue_patterns:
            try:
                # 创建IssuePattern记录
                pattern_id = self.db_service.create_issue_pattern(pattern)
                
                # 同步到向量数据库
                self.sync_service.sync_issue_pattern(pattern_id)
                
                success_count += 1
                print(f"✓ 成功导入模式 {pattern['title']}")
                
            except Exception as e:
                print(f"✗ 导入失败 {pattern['title']}: {e}")
        
        print(f"导入完成: {success_count}/{len(issue_patterns)} 个模式成功导入")
    
    def run_integration(self, sample_size: int = 50):
        """运行集成流程"""
        print("=== BigVul数据集与MAS系统集成 ===")
        
        # 1. 加载数据
        df = self.load_bigvul_data()
        if df is None:
            return
        
        # 2. 转换为IssuePattern格式
        issue_patterns = self.convert_to_issue_patterns(df, sample_size)
        
        # 3. 导入数据库
        self.import_to_database(issue_patterns)
        
        print("\n=== 集成完成 ===")

def main():
    """主函数"""
    integrator = BigVulIntegration()
    integrator.run_integration(sample_size=50)

if __name__ == "__main__":
    main()
