#!/usr/bin/env python3
"""
BigVul数据集下载脚本
自动下载和处理BigVul数据集到本地目录
修复SSL证书验证问题，添加重试机制和备用下载源
"""

import os
import requests
import pandas as pd
import zipfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib
import time
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import ssl
import subprocess

# 禁用SSL警告
urllib3.disable_warnings(InsecureRequestWarning)

class BigVulDownloader:
    """BigVul数据集下载器"""
    
    def __init__(self, base_dir: str = "e:\\MyOwn\\ProgramStudy\\MAS\\tests\\BigVul"):
        self.base_dir = Path(base_dir)
        
        # 备用下载源列表
        self.data_sources = {
            "bigvul_main": {
                "urls": [
                    "https://raw.githubusercontent.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/master/BigVul.csv",
                    "https://gist.githubusercontent.com/ZeoVan/raw/master/BigVul.csv",
                    "https://cdn.jsdelivr.net/gh/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset@master/BigVul.csv"
                ],
                "filename": "BigVul.csv",
                "description": "主要BigVul数据集（CSV格式）",
                "md5": None
            },
            "bigvul_processed": {
                "urls": [
                    "https://raw.githubusercontent.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/master/processed_func.csv",
                    "https://gist.githubusercontent.com/ZeoVan/raw/master/processed_func.csv",
                    "https://cdn.jsdelivr.net/gh/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset@master/processed_func.csv"
                ],
                "filename": "processed_func.csv",
                "description": "处理后的函数级数据",
                "md5": None
            },
            "bigvul_metadata": {
                "urls": [
                    "https://raw.githubusercontent.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/master/big-vul.csv",
                    "https://gist.githubusercontent.com/ZeoVan/raw/master/big-vul.csv",
                    "https://cdn.jsdelivr.net/gh/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset@master/big-vul.csv"
                ],
                "filename": "big-vul.csv",
                "description": "BigVul元数据",
                "md5": None
            }
        }
        
        # 创建目录结构
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            "raw",           # 原始数据
            "processed",     # 处理后的数据
            "metadata",      # 元数据
            "samples",       # 样本数据
            "validation"     # 验证数据
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {dir_path}")
    
    def download_dataset(self, source_key: str = "all") -> bool:
        """下载数据集"""
        if source_key == "all":
            success = True
            for key in self.data_sources.keys():
                if not self._download_single_source(key):
                    success = False
            return success
        else:
            return self._download_single_source(source_key)
    
    def _download_single_source(self, source_key: str) -> bool:
        """下载单个数据源（带重试机制）"""
        if source_key not in self.data_sources:
            print(f"错误: 未知的数据源 '{source_key}'")
            return False
        
        source = self.data_sources[source_key]
        filename = source["filename"]
        file_path = self.base_dir / "raw" / filename
        
        print(f"正在下载: {source['description']}")
        
        # 如果文件已存在，跳过下载
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > 0:
                print(f"✓ 文件已存在，跳过下载: {file_path} ({file_size} bytes)")
                return True
            else:
                print(f"⚠ 文件大小为0，重新下载: {file_path}")
                file_path.unlink(missing_ok=True)
        
        # 尝试多个URL和重试机制
        max_retries = 3
        retry_delay = 5  # 秒
        
        for url in source["urls"]:
            print(f"尝试URL: {url}")
            print(f"保存到: {file_path}")
            
            for attempt in range(max_retries):
                try:
                    # 方法1: 禁用SSL验证
                    response = requests.get(url, stream=True, verify=False, timeout=30)
                    response.raise_for_status()
                    
                    # 保存文件
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # 验证文件大小
                    file_size = file_path.stat().st_size
                    if file_size > 0:
                        print(f"✓ 下载成功: {filename} ({file_size} bytes)")
                        return True
                    else:
                        print(f"⚠ 文件大小为0，删除并重试...")
                        file_path.unlink(missing_ok=True)
                        
                except requests.exceptions.SSLError as e:
                    print(f"SSL错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                    
                    # 方法2: 尝试使用certifi证书包
                    try:
                        import certifi
                        response = requests.get(url, stream=True, verify=certifi.where(), timeout=30)
                        response.raise_for_status()
                        
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        file_size = file_path.stat().st_size
                        if file_size > 0:
                            print(f"✓ 使用certifi下载成功: {filename} ({file_size} bytes)")
                            return True
                            
                    except Exception as e2:
                        print(f"certifi方法也失败: {e2}")
                        file_path.unlink(missing_ok=True)
                
                except requests.exceptions.RequestException as e:
                    print(f"网络错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                except Exception as e:
                    print(f"未知错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                # 重试前等待
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
        
        print(f"❌ 所有URL和重试都失败: {filename}")
        return False
    
    def download_with_alternative_methods(self) -> bool:
        """使用替代方法下载（如果主要方法失败）"""
        print("\n=== 尝试替代下载方法 ===")
        
        # 方法1: 使用wget（如果可用）
        if self._try_wget_download():
            return True
        
        # 方法2: 使用curl（如果可用）
        if self._try_curl_download():
            return True
        
        # 方法3: 创建手动下载指南
        self._create_manual_download_guide()
        return False
    
    def _try_wget_download(self) -> bool:
        """尝试使用wget下载"""
        try:
            for source_key, source in self.data_sources.items():
                filename = source["filename"]
                file_path = self.base_dir / "raw" / filename
                
                if file_path.exists() and file_path.stat().st_size > 0:
                    continue
                
                for url in source["urls"]:
                    try:
                        print(f"尝试使用wget下载: {filename}")
                        result = subprocess.run(
                            ["wget", "--no-check-certificate", "-O", str(file_path), url],
                            capture_output=True, text=True, timeout=60
                        )
                        
                        if result.returncode == 0 and file_path.exists() and file_path.stat().st_size > 0:
                            print(f"✓ wget下载成功: {filename}")
                            return True
                            
                    except Exception as e:
                        print(f"wget下载失败: {e}")
                        
        except Exception as e:
            print(f"wget方法不可用: {e}")
            
        return False
    
    def _try_curl_download(self) -> bool:
        """尝试使用curl下载"""
        try:
            for source_key, source in self.data_sources.items():
                filename = source["filename"]
                file_path = self.base_dir / "raw" / filename
                
                if file_path.exists() and file_path.stat().st_size > 0:
                    continue
                
                for url in source["urls"]:
                    try:
                        print(f"尝试使用curl下载: {filename}")
                        result = subprocess.run(
                            ["curl", "-k", "-L", "-o", str(file_path), url],
                            capture_output=True, text=True, timeout=60
                        )
                        
                        if result.returncode == 0 and file_path.exists() and file_path.stat().st_size > 0:
                            print(f"✓ curl下载成功: {filename}")
                            return True
                            
                    except Exception as e:
                        print(f"curl下载失败: {e}")
                        
        except Exception as e:
            print(f"curl方法不可用: {e}")
            
        return False
    
    def _create_manual_download_guide(self):
        """创建手动下载指南"""
        print("\n=== 手动下载指南 ===")
        print("如果自动下载失败，请手动下载以下文件：")
        
        guide_file = self.base_dir / "manual_download_guide.txt"
        with open(guide_file, "w", encoding="utf-8") as f:
            f.write("BigVul数据集手动下载指南\n")
            f.write("=" * 50 + "\n\n")
            
            for source_key, source in self.data_sources.items():
                f.write(f"文件: {source['filename']}\n")
                f.write(f"描述: {source['description']}\n")
                f.write("下载链接:\n")
                
                for i, url in enumerate(source["urls"], 1):
                    f.write(f"  {i}. {url}\n")
                
                f.write(f"保存到: {self.base_dir / 'raw' / source['filename']}\n")
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"手动下载指南已保存到: {guide_file}")
    
    def load_and_preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """加载和预处理数据"""
        datasets = {}
        
        # 检查文件是否存在
        main_file = self.base_dir / "raw" / "BigVul.csv"
        if not main_file.exists():
            print("主要数据集文件不存在，请先下载")
            return datasets
        
        print("加载主要数据集...")
        try:
            df_main = pd.read_csv(main_file)
            datasets["main"] = df_main
            print(f"主要数据集大小: {df_main.shape}")
            
            # 保存处理后的版本
            processed_path = self.base_dir / "processed" / "bigvul_processed.csv"
            self._preprocess_data(df_main).to_csv(processed_path, index=False)
            print(f"处理后的数据保存到: {processed_path}")
            
        except Exception as e:
            print(f"加载主要数据集失败: {e}")
        
        return datasets
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        processed_df = df.copy()
        
        # 数据清洗
        if 'target' in processed_df.columns:
            processed_df = processed_df.dropna(subset=['target'])
        
        # 添加中文漏洞类型映射
        cwe_mapping = {
            'CWE-119': '缓冲区错误',
            'CWE-125': '越界读取',
            'CWE-787': '越界写入',
            'CWE-20': '输入验证不当',
            'CWE-79': '跨站脚本',
            'CWE-89': 'SQL注入',
            'CWE-94': '代码注入',
            'CWE-400': '资源耗尽',
            'CWE-476': '空指针引用',
            'CWE-502': '反序列化漏洞'
        }
        
        if 'vulnerability' in processed_df.columns:
            processed_df['vul_type_chinese'] = processed_df['vulnerability'].map(cwe_mapping)
        
        return processed_df

def main():
    """主函数"""
    downloader = BigVulDownloader()
    
    print("=== BigVul数据集下载工具（增强版） ===")
    print(f"目标目录: {downloader.base_dir}")
    print()
    
    # 下载数据集
    print("1. 下载数据集...")
    success = downloader.download_dataset("all")
    
    if not success:
        print("\n主要下载方法失败，尝试替代方法...")
        success = downloader.download_with_alternative_methods()
    
    if success:
        print("\n2. 加载和预处理数据...")
        datasets = downloader.load_and_preprocess_data()
        
        print("\n=== 下载和预处理完成 ===")
        
        # 显示数据集统计信息
        if datasets:
            print("\n数据集统计:")
            for name, df in datasets.items():
                if df is not None:
                    print(f"  {name}: {df.shape}")
    else:
        print("\n所有下载方法都失败，请检查网络连接或使用手动下载指南")

if __name__ == "__main__":
    main()