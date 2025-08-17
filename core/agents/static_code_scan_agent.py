import subprocess
import json
import os
import time
from typing import Dict, Any, List
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService
from infrastructure.config.settings import STATIC_TOOLS_CONFIG

class StaticCodeScanAgent(BaseAgent):
    """静态代码扫描智能体 - 集成多种静态分析工具"""
    
    def __init__(self):
        super().__init__("static_scan_agent", "静态代码扫描智能体")
        self.db_service = DatabaseService()
        self.tools_config = STATIC_TOOLS_CONFIG
        self.supported_tools = {
            "pylint": self._run_pylint,
            "bandit": self._run_bandit,
            "flake8": self._run_flake8,
            "mypy": self._run_mypy,
            "safety": self._run_safety
        }
        
    async def handle_message(self, message: Message):
        """处理静态扫描请求"""
        if message.message_type == "static_scan_request":
            requirement_id = message.content.get("requirement_id")
            task_data = message.content.get("task_data")
            
            print(f"🔍 开始静态代码扫描 - 需求ID: {requirement_id}")
            
            result = await self.execute_task({
                "requirement_id": requirement_id,
                **task_data
            })
            
            # 转发结果给汇总智能体
            await self.send_message(
                receiver="summary_agent",
                content={
                    "requirement_id": requirement_id,
                    "analysis_type": "static_code_scan",
                    "result": result
                },
                message_type="analysis_result"
            )
            
            print(f"✅ 静态代码扫描完成 - 需求ID: {requirement_id}")
            
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行静态代码扫描任务"""
        code_directory = task_data.get("code_directory")
        requirement_id = task_data.get("requirement_id")
        start_time = time.time()
        
        if not code_directory or not os.path.exists(code_directory):
            error_msg = f"无效的代码目录: {code_directory}"
            print(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}
            
        # 执行各种静态分析工具
        scan_results = {
            "tools_results": {},
            "summary": {
                "total_issues": 0,
                "critical_issues": 0,
                "security_issues": 0,
                "style_issues": 0,
                "type_issues": 0,
                "dependency_issues": 0
            },
            "recommendations": [],
            "status": "completed",
            "scan_time": 0
        }
        
        try:
            # Pylint - 代码质量检查
            if self.tools_config.get("pylint", {}).get("enabled", False):
                print("🔍 运行 Pylint 代码质量检查...")
                pylint_result = await self._run_pylint(code_directory)
                scan_results["tools_results"]["pylint"] = pylint_result
                
            # Bandit - 安全漏洞扫描
            if self.tools_config.get("bandit", {}).get("enabled", False):
                print("🔒 运行 Bandit 安全扫描...")
                bandit_result = await self._run_bandit(code_directory)
                scan_results["tools_results"]["bandit"] = bandit_result
                
            # Flake8 - 代码风格检查
            if self.tools_config.get("flake8", {}).get("enabled", False):
                print("📝 运行 Flake8 风格检查...")
                flake8_result = await self._run_flake8(code_directory)
                scan_results["tools_results"]["flake8"] = flake8_result
                
            # MyPy - 类型检查
            if self.tools_config.get("mypy", {}).get("enabled", False):
                print("🔤 运行 MyPy 类型检查...")
                mypy_result = await self._run_mypy(code_directory)
                scan_results["tools_results"]["mypy"] = mypy_result
                
            # Safety - 依赖安全检查
            if self.tools_config.get("safety", {}).get("enabled", False):
                print("📦 运行 Safety 依赖安全检查...")
                safety_result = await self._run_safety(code_directory)
                scan_results["tools_results"]["safety"] = safety_result
                
            # 分析扫描结果
            analyzed_results = await self._analyze_scan_results(scan_results)
            
            # 计算处理时间
            processing_time = int(time.time() - start_time)
            analyzed_results["scan_time"] = processing_time
            
            # 保存结果到数据库
            await self.db_service.save_analysis_result(
                requirement_id=requirement_id,
                agent_type="static_code_scan",
                result_data=analyzed_results,
                status="completed"
            )
            
            print(f"⏱️  静态扫描完成，耗时 {processing_time} 秒")
            return analyzed_results
            
        except Exception as e:
            processing_time = int(time.time() - start_time)
            error_msg = f"静态代码扫描失败: {e}"
            print(f"❌ {error_msg}")
            
            # 保存错误结果
            await self.db_service.save_analysis_result(
                requirement_id=requirement_id,
                agent_type="static_code_scan",
                result_data={"status": "error", "error": str(e)},
                status="error"
            )
            
            return {"status": "error", "error": error_msg, "scan_time": processing_time}
        
    async def _run_pylint(self, directory: str) -> Dict[str, Any]:
        """运行Pylint代码质量检查"""
        try:
            config = self.tools_config.get("pylint", {})
            cmd = ["pylint"] + config.get("args", []) + [directory]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=config.get("timeout", 60)
            )
            
            if result.stdout:
                try:
                    # 尝试解析JSON输出
                    output = json.loads(result.stdout)
                    return {"status": "success", "output": output}
                except json.JSONDecodeError:
                    # 解析文本输出
                    issues = self._parse_pylint_text_output(result.stdout)
                    return {"status": "success", "output": issues, "raw_output": result.stdout}
            else:
                return {"status": "no_issues", "output": []}
                
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Pylint 扫描超时"}
        except FileNotFoundError:
            return {"status": "not_found", "error": "Pylint 工具未安装"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    def _parse_pylint_text_output(self, output: str) -> List[Dict[str, Any]]:
        """解析Pylint文本输出"""
        issues = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if ':' in line and ('error' in line.lower() or 'warning' in line.lower()):
                parts = line.split(':')
                if len(parts) >= 4:
                    issues.append({
                        "path": parts[0],
                        "line": parts[1],
                        "column": parts[2] if len(parts) > 2 else "0",
                        "type": "error" if "error" in line.lower() else "warning",
                        "message": ':'.join(parts[3:]).strip()
                    })
        return issues
            
    async def _run_bandit(self, directory: str) -> Dict[str, Any]:
        """运行Bandit安全扫描"""
        try:
            config = self.tools_config.get("bandit", {})
            cmd = ["bandit", "-r", directory] + config.get("args", [])
            
            result = subprocess.run(
                cmd,
                capture_output=True, 
                text=True,
                timeout=config.get("timeout", 30)
            )
            
            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                    return {"status": "success", "output": output}
                except json.JSONDecodeError:
                    return {
                        "status": "success",
                        "output": {"results": []},
                        "raw_output": result.stdout
                    }
            else:
                return {"status": "no_issues", "output": {"results": []}}
                
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Bandit 扫描超时"}
        except FileNotFoundError:
            return {"status": "not_found", "error": "Bandit 工具未安装"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _run_flake8(self, directory: str) -> Dict[str, Any]:
        """运行Flake8代码风格检查"""
        try:
            config = self.tools_config.get("flake8", {})
            cmd = ["flake8", directory] + config.get("args", [])
            
            result = subprocess.run(
                cmd,
                capture_output=True, 
                text=True,
                timeout=config.get("timeout", 30)
            )
            
            issues = []
            if result.stdout:
                # Flake8输出需要手动解析
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            issues.append({
                                "file": parts[0],
                                "line": parts[1],
                                "column": parts[2],
                                "code": parts[3].split()[0] if parts[3].split() else "",
                                "message": ' '.join(parts[3].split()[1:]) if len(parts[3].split()) > 1 else parts[3]
                            })
                            
            return {"status": "success", "output": issues}
            
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Flake8 扫描超时"}
        except FileNotFoundError:
            return {"status": "not_found", "error": "Flake8 工具未安装"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _run_mypy(self, directory: str) -> Dict[str, Any]:
        """运行MyPy类型检查"""
        try:
            config = self.tools_config.get("mypy", {})
            cmd = ["mypy", directory] + config.get("args", ["--ignore-missing-imports"])
            
            result = subprocess.run(
                cmd,
                capture_output=True, 
                text=True,
                timeout=config.get("timeout", 60)
            )
            
            issues = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line and ('error' in line.lower() or 'note' in line.lower()):
                        parts = line.split(':', 3)
                        if len(parts) >= 3:
                            issues.append({
                                "file": parts[0],
                                "line": parts[1],
                                "type": "type_error",
                                "message": ':'.join(parts[2:]).strip()
                            })
                            
            return {"status": "success", "output": issues}
            
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "MyPy 扫描超时"}
        except FileNotFoundError:
            return {"status": "not_found", "error": "MyPy 工具未安装"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _run_safety(self, directory: str) -> Dict[str, Any]:
        """运行Safety依赖安全检查"""
        try:
            config = self.tools_config.get("safety", {})
            
            # 查找requirements文件
            req_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file in ['requirements.txt', 'requirements-dev.txt', 'Pipfile']:
                        req_files.append(os.path.join(root, file))
                        
            if not req_files:
                return {"status": "no_requirements", "output": {"vulnerabilities": []}}
                
            cmd = ["safety", "check"] + config.get("args", ["--json"])
            if req_files:
                cmd.extend(["-r", req_files[0]])
            
            result = subprocess.run(
                cmd,
                capture_output=True, 
                text=True,
                timeout=config.get("timeout", 30)
            )
            
            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                    return {"status": "success", "output": output}
                except json.JSONDecodeError:
                    return {
                        "status": "success",
                        "output": {"vulnerabilities": []},
                        "raw_output": result.stdout
                    }
            else:
                return {"status": "no_issues", "output": {"vulnerabilities": []}}
                
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Safety 扫描超时"}
        except FileNotFoundError:
            return {"status": "not_found", "error": "Safety 工具未安装"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _analyze_scan_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析扫描结果并生成汇总"""
        summary = results["summary"]
        tools_results = results["tools_results"]
        recommendations = []
        
        # 统计Pylint问题
        if "pylint" in tools_results and tools_results["pylint"].get("status") == "success":
            pylint_issues = tools_results["pylint"]["output"]
            if isinstance(pylint_issues, list):
                for issue in pylint_issues:
                    summary["total_issues"] += 1
                    if issue.get("type") in ["error", "fatal"]:
                        summary["critical_issues"] += 1
                        
        # 统计Bandit安全问题
        if "bandit" in tools_results and tools_results["bandit"].get("status") == "success":
            bandit_output = tools_results["bandit"]["output"]
            if isinstance(bandit_output, dict) and "results" in bandit_output:
                security_issues = len(bandit_output["results"])
                summary["security_issues"] = security_issues
                summary["total_issues"] += security_issues
                
        # 统计Flake8风格问题
        if "flake8" in tools_results and tools_results["flake8"].get("status") == "success":
            flake8_issues = len(tools_results["flake8"]["output"])
            summary["style_issues"] = flake8_issues
            summary["total_issues"] += flake8_issues
            
        # 统计MyPy类型问题
        if "mypy" in tools_results and tools_results["mypy"].get("status") == "success":
            mypy_issues = len(tools_results["mypy"]["output"])
            summary["type_issues"] = mypy_issues
            summary["total_issues"] += mypy_issues
            
        # 统计Safety依赖问题
        if "safety" in tools_results and tools_results["safety"].get("status") == "success":
            safety_output = tools_results["safety"]["output"]
            if isinstance(safety_output, dict) and "vulnerabilities" in safety_output:
                dep_issues = len(safety_output["vulnerabilities"])
                summary["dependency_issues"] = dep_issues
                summary["total_issues"] += dep_issues
                
        # 生成建议
        if summary["critical_issues"] > 0:
            recommendations.append({
                "type": "critical",
                "message": f"发现 {summary['critical_issues']} 个严重代码问题，建议优先修复",
                "priority": "high"
            })
            
        if summary["security_issues"] > 0:
            recommendations.append({
                "type": "security",
                "message": f"发现 {summary['security_issues']} 个安全漏洞，建议立即处理",
                "priority": "critical"
            })
            
        if summary["type_issues"] > 5:
            recommendations.append({
                "type": "type_safety",
                "message": f"发现 {summary['type_issues']} 个类型问题，建议添加类型注解",
                "priority": "medium"
            })
            
        if summary["style_issues"] > 10:
            recommendations.append({
                "type": "style",
                "message": f"发现 {summary['style_issues']} 个代码风格问题，建议统一代码规范",
                "priority": "medium"
            })
            
        if summary["dependency_issues"] > 0:
            recommendations.append({
                "type": "dependencies",
                "message": f"发现 {summary['dependency_issues']} 个依赖安全问题，建议更新依赖包",
                "priority": "high"
            })
            
        if summary["total_issues"] == 0:
            recommendations.append({
                "type": "clean",
                "message": "代码质量良好，未发现明显问题",
                "priority": "info"
            })
            
        results["recommendations"] = recommendations
        return results