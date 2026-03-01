#!/usr/bin/env python3
"""
BigVul数据集验证脚本
验证数据集完整性并生成统计报告
"""

import pandas as pd
from pathlib import Path
import json

class BigVulValidator:
    """BigVul数据集验证器"""
    
    def __init__(self, base_dir: str = "e:\\MyOwn\\ProgramStudy\\MAS\\tests\\BigVul"):
        self.base_dir = Path(base_dir)
    
    def validate_dataset(self) -> Dict:
        """验证数据集完整性"""
        validation_results = {}
        
        # 检查文件存在性
        files_to_check = [
            "raw/BigVul.csv",
            "raw/processed_func.csv", 
            "raw/big-vul.csv",
            "processed/bigvul_processed.csv"
        ]
        
        for file_path in files_to_check:
            full_path = self.base_dir / file_path
            validation_results[file_path] = {
                "exists": full_path.exists(),
                "size": full_path.stat().st_size if full_path.exists() else 0
            }
        
        # 加载并验证数据质量
        if (self.base_dir / "processed/bigvul_processed.csv").exists():
            try:
                df = pd.read_csv(self.base_dir / "processed/bigvul_processed.csv")
                validation_results["data_quality"] = {
                    "total_records": len(df),
                    "columns": list(df.columns),
                    "null_counts": df.isnull().sum().to_dict(),
                    "vulnerable_count": df['target'].sum() if 'target' in df.columns else 0,
                    "safe_count": len(df) - df['target'].sum() if 'target' in df.columns else 0
                }
            except Exception as e:
                validation_results["data_quality"] = {"error": str(e)}
        
        return validation_results
    
    def generate_report(self, validation_results: Dict) -> str:
        """生成验证报告"""
        report = "=== BigVul数据集验证报告 ===\n\n"
        
        # 文件存在性检查
        report += "1. 文件存在性检查:\n"
        for file_path, result in validation_results.items():
            if file_path != "data_quality":
                status = "✓ 存在" if result["exists"] else "✗ 缺失"
                size_info = f" ({result['size']} bytes)" if result["exists"] else ""
                report += f"   {file_path}: {status}{size_info}\n"
        
        # 数据质量检查
        if "data_quality" in validation_results:
            report += "\n2. 数据质量检查:\n"
            quality = validation_results["data_quality"]
            
            if "error" in quality:
                report += f"   错误: {quality['error']}\n"
            else:
                report += f"   总记录数: {quality['total_records']}\n"
                report += f"   漏洞样本数: {quality['vulnerable_count']}\n"
                report += f"   安全样本数: {quality['safe_count']}\n"
                report += f"   列数量: {len(quality['columns'])}\n"
        
        return report

def main():
    """主验证函数"""
    validator = BigVulValidator()
    
    print("开始验证BigVul数据集...")
    results = validator.validate_dataset()
    report = validator.generate_report(results)
    
    print(report)
    
    # 保存报告
    report_path = validator.base_dir / "validation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"验证报告已保存到: {report_path}")

if __name__ == "__main__":
    main()
