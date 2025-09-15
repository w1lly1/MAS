#!/usr/bin/env python3
"""
报告管理器使用示例
演示如何使用统一的报告管理系统
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from infrastructure.reports import report_manager

def demo_report_generation():
    """演示报告生成功能"""
    print("🚀 开始演示报告管理器功能...\n")
    
    try:
        # 1. 生成分析报告
        analysis_data = {
            "project": "MAS Multi-Agent System",
            "timestamp": "2024-01-15 10:30:00",
            "code_quality": {
                "overall_score": 8.5,
                "issues_found": 3,
                "recommendations": [
                    "优化异常处理",
                    "添加更多单元测试",
                    "改进代码注释"
                ]
            },
            "performance": {
                "response_time": "150ms",
                "memory_usage": "2.1GB",
                "cpu_usage": "15%"
            }
        }
        
        analysis_report = report_manager.generate_analysis_report(analysis_data)
        
        # 2. 生成兼容性报告
        compatibility_data = {
            "系统兼容性检查": {
                "Python版本": "3.12.3 ✅",
                "Transformers": "4.56.0 ✅", 
                "Torch": "2.8.0+cu128 ✅",
                "模型兼容性": "Qwen1.5-7B-Chat ✅"
            },
            "已解决问题": [
                "ChatGLM2-6B padding_side 兼容性问题",
                "Transformers 4.56.0 版本兼容",
                "模型加载优化"
            ],
            "系统状态": "生产环境就绪 ✅"
        }
        
        compatibility_report = report_manager.generate_compatibility_report(compatibility_data)
        
        # 3. 生成部署报告
        deployment_content = """# MAS 系统部署报告

## 部署概要
- **部署时间**: 2024-01-15 10:30:00
- **系统版本**: v2.0.0
- **部署环境**: Linux Production

## 系统健康检查
- **AI模型**: Qwen1.5-7B-Chat 运行正常
- **数据库**: SQLite 连接正常
- **API接口**: 响应时间 < 200ms

## 部署状态
🟢 **系统状态**: 运行正常
"""
        
        deployment_report = report_manager.generate_deployment_report(deployment_content)
        
        # 4. 列出所有报告
        print("\n📋 当前报告文件列表:")
        all_reports = report_manager.list_reports()
        for report_type, files in all_reports.items():
            if files:  # 只显示有文件的目录
                print(f"\n📁 {report_type.upper()}:")
                for file_path in files:
                    print(f"  - {file_path.name}")
        
        print("\n🎉 报告管理器演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_report_generation()
