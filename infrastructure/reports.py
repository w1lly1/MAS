"""
报告生成和管理工具
统一管理所有系统生成的报告文件
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ReportManager:
    """报告管理器"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # 使用与core目录同级的reports文件夹
            project_root = Path(__file__).parent.parent
            base_dir = project_root / "reports"
        
        self.base_dir = Path(base_dir)
        self.directories = {
            "analysis": self.base_dir / "analysis",
            "compatibility": self.base_dir / "compatibility",
            "deployment": self.base_dir / "deployment", 
            "testing": self.base_dir / "testing"
        }
        
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有报告目录存在"""
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_analysis_report(self, content: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """生成代码分析报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{timestamp}.json"
        
        report_path = self.directories["analysis"] / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        print(f"📊 分析报告已生成: {report_path}")
        return report_path
    
    def generate_compatibility_report(self, content: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """生成兼容性报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compatibility_report_{timestamp}.md"
        
        report_path = self.directories["compatibility"] / filename
        
        if isinstance(content, dict):
            # 如果是字典，转换为Markdown格式
            markdown_content = self._dict_to_markdown(content)
        else:
            markdown_content = str(content)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"🔧 兼容性报告已生成: {report_path}")
        return report_path
    
    def generate_deployment_report(self, content: str, filename: Optional[str] = None) -> Path:
        """生成部署报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deployment_report_{timestamp}.md"
        
        report_path = self.directories["deployment"] / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"🚀 部署报告已生成: {report_path}")
        return report_path
    
    def generate_testing_report(self, content: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """生成测试报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"testing_report_{timestamp}.json"
        
        report_path = self.directories["testing"] / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        print(f"🧪 测试报告已生成: {report_path}")
        return report_path
    
    def _dict_to_markdown(self, data: Dict[str, Any], level: int = 1) -> str:
        """将字典转换为Markdown格式"""
        lines = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{'#' * level} {key}\n")
                lines.append(self._dict_to_markdown(value, level + 1))
            elif isinstance(value, list):
                lines.append(f"{'#' * level} {key}\n")
                for item in value:
                    lines.append(f"- {item}")
                lines.append("")
            else:
                lines.append(f"**{key}**: {value}\n")
        
        return "\n".join(lines)
    
    def list_reports(self, report_type: Optional[str] = None) -> Dict[str, list]:
        """列出所有报告文件"""
        reports = {}
        
        if report_type:
            if report_type in self.directories:
                dir_path = self.directories[report_type]
                reports[report_type] = list(dir_path.glob("*"))
        else:
            for report_type, dir_path in self.directories.items():
                reports[report_type] = list(dir_path.glob("*"))
        
        return reports
    
    def cleanup_old_reports(self, days: int = 30):
        """清理旧的报告文件"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        removed_count = 0
        
        for report_type, dir_path in self.directories.items():
            for file_path in dir_path.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
        
        print(f"🧹 已清理 {removed_count} 个超过 {days} 天的报告文件")

# 全局报告管理器实例
report_manager = ReportManager()
