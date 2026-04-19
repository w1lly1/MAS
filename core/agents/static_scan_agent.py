import os
import subprocess
import json
import ast
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple, Set
from .base_agent import BaseAgent, Message
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG
from infrastructure.reports import report_manager
from utils import log, LogLevel

class StaticCodeScanAgent(BaseAgent):
    """传统静态代码扫描智能体 - 使用专业静态分析工具"""
    
    def __init__(self):
        super().__init__("static_scan_agent", "静态代码扫描智能体")
        self.db_service = DatabaseService()
        
        # 从统一配置获取
        from infrastructure.config.ai_agents import get_ai_agent_config
        self.agent_config = get_ai_agent_config().get_static_scan_agent_config()
        
        # 静态分析工具配置
        self.static_tools = {
            "python": {
                "linters": ["pylint", "flake8", "pycodestyle", "pydocstyle"],
                "security": ["bandit", "safety"],
                "complexity": ["radon", "mccabe"],
                "type_check": ["mypy"]
            },
            "javascript": {
                "linters": ["eslint", "jshint"],
                "security": ["eslint-plugin-security"],
                "complexity": ["complexity-report"]
            },
            "java": {
                "linters": ["checkstyle", "pmd"],
                "security": ["spotbugs"],
                "complexity": ["metrics"]
            }
        }
        
        # 工具可用性状态
        self.available_tools = {}
        self._processed_requests: Set[tuple] = set()  # (requirement_id, run_id)
        
    async def initialize(self):
        """初始化静态分析工具"""
        await super().initialize()
        await self._check_tool_availability()
        
    async def _check_tool_availability(self):
        """检查静态分析工具的可用性"""
        log("static_scan_tools", LogLevel.INFO, "🔧 检查静态分析工具可用性...")
        
        tools_to_check = [
            "pylint", "flake8", "bandit", "radon", "mypy", "semgrep", "cppcheck", "clang-tidy", "spotbugs"
        ]
        
        for tool in tools_to_check:
            try:
                if shutil.which(tool) is None:
                    self.available_tools[tool] = False
                    log("static_scan_tools", LogLevel.WARNING, f"⚠️ {tool} 未安装")
                    continue
                check_timeout = self.agent_config.get("tool_check_timeout", 5)
                result = subprocess.run([tool, "--version"], 
                                      capture_output=True, text=True, timeout=check_timeout)
                if result.returncode == 0:
                    self.available_tools[tool] = True
                    log("static_scan_tools", LogLevel.INFO, f"✅ {tool} 可用")
                else:
                    self.available_tools[tool] = False
                    log("static_scan_tools", LogLevel.WARNING, f"⚠️ {tool} 不可用")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.available_tools[tool] = False
                log("static_scan_tools", LogLevel.WARNING, f"⚠️ {tool} 未安装")
        
        log("static_scan_tools", LogLevel.INFO, f"📊 可用工具: {[k for k, v in self.available_tools.items() if v]}")
        
    async def handle_message(self, message: Message):
        """处理静态代码扫描请求"""
        if message.message_type == "static_scan_request":
            requirement_id = message.content.get("requirement_id")
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            file_path = message.content.get("file_path")
            run_id = message.content.get('run_id')
            readable_file = message.content.get('readable_file')
            key = (requirement_id, run_id)
            if key in self._processed_requests:
                log("static_scan_tools", LogLevel.INFO, f"🧪 [StaticScan] 跳过重复扫描 requirement={requirement_id} run_id={run_id}")
                return
            self._processed_requests.add(key)
            log("static_scan_tools", LogLevel.INFO, f"🧪 [StaticScan] 开始扫描 requirement={requirement_id} run_id={run_id} file={file_path}")
            
            # 执行传统静态分析
            result = await self._traditional_static_analysis(code_content, code_directory)
            # enrich with file path and run context
            if file_path:
                result['file_path'] = file_path
            if readable_file:
                result['readable_file'] = readable_file
            if run_id:
                result['run_id'] = run_id
                try:
                    agent_payload = {
                        "requirement_id": requirement_id,
                        "file_path": file_path,
                        "run_id": run_id,
                        "static_scan_result": result,
                        "generated_at": self._get_current_time()
                    }
                    report_manager.generate_run_scoped_report(run_id, agent_payload, f"static_req_{requirement_id}.json", subdir="agents/static")
                except Exception as e:
                    log("static_scan_tools", LogLevel.WARNING, f"⚠️ 静态扫描Agent单独报告生成失败 requirement={requirement_id} run_id={run_id}: {e}")
            log("static_scan_tools", LogLevel.INFO, f"🧪 [StaticScan] 完成 requirement={requirement_id} issues_total={result.get('summary',{}).get('total_issues')} run_id={run_id}")
            await self.dispatch_message(
                receiver="ai_code_quality_agent",
                content={
                    "requirement_id": requirement_id,
                    "static_scan_results": result,
                    "code_content": code_content,
                    "code_directory": code_directory,
                    "file_path": file_path,
                    "run_id": run_id
                },
                message_type="static_scan_complete"
            )
            # send to summary
            await self.dispatch_message(
                receiver="summary_agent",
                content={
                    "requirement_id": requirement_id,
                    "analysis_type": "static_analysis",
                    "result": result,
                    "file_path": file_path,
                    "readable_file": readable_file,
                    "run_id": run_id
                },
                message_type="analysis_result"
            )
            log("static_scan_tools", LogLevel.INFO, f"✅ 静态代码扫描完成,结果已发送 requirement={requirement_id} run_id={run_id}")
            
    async def _traditional_static_analysis(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """传统静态代码分析"""
        
        try:
            log("static_scan_tools", LogLevel.INFO, "🔍 执行传统静态代码分析...")
            
            # 1. 语言检测
            language = self._detect_language(code_content)
            log("static_scan_tools", LogLevel.INFO, f"📝 检测到语言: {language}")
            
            # 2. 基础代码结构分析
            code_structure = await self._analyze_code_structure(code_content, language)
            
            # 3. 代码质量检查(pylint, flake8等)
            quality_issues = await self._run_quality_checks(code_content, code_directory, language)
            
            # 4. 安全漏洞扫描(bandit等)
            security_issues = await self._run_security_scans(code_content, code_directory, language)
            
            # 5. 复杂度分析(radon等)
            complexity_analysis = await self._run_complexity_analysis(code_content, code_directory, language)
            
            # 6. 类型检查(mypy等)
            type_issues = await self._run_type_checks(code_content, code_directory, language)
            
            # 7. 代码风格检查
            style_issues = await self._run_style_checks(code_content, code_directory, language)
            
            # 8. 综合统计
            summary = await self._generate_scan_summary(
                quality_issues, security_issues, complexity_analysis, type_issues, style_issues
            )
            
            log("static_scan_tools", LogLevel.INFO, "✅ 传统静态分析完成")
            
            return {
                "scan_type": "traditional_static_analysis",
                "language": language,
                "code_structure": code_structure,
                "quality_issues": quality_issues,
                "security_issues": security_issues,
                "complexity_analysis": complexity_analysis,
                "type_issues": type_issues,
                "style_issues": style_issues,
                "summary": summary,
                "tools_used": [tool for tool, available in self.available_tools.items() if available],
                "scan_timestamp": self._get_current_time(),
                "scan_status": "completed"
            }
            
        except Exception as e:
            log("static_scan_tools", LogLevel.ERROR, f"❌ 静态分析过程中出错: {e}")
            return {
                "scan_type": "traditional_static_analysis",
                "error": str(e),
                "scan_status": "failed"
            }
        
    async def _analyze_code_structure(self, code_content: str, language: str) -> Dict[str, Any]:
        """分析代码结构"""
        structure = {
            "language": language,
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "functions": [],
            "classes": [],
            "imports": [],
            "constants": []
        }
        
        try:
            lines = code_content.split('\n')
            structure["total_lines"] = len(lines)
            
            # 基础统计
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    structure["blank_lines"] += 1
                elif stripped.startswith('#') or stripped.startswith('//'):
                    structure["comment_lines"] += 1
                else:
                    structure["code_lines"] += 1
            
            # 语言特定结构分析
            if language == "python":
                structure.update(await self._parse_python_structure(code_content))
            elif language in ["javascript", "typescript"]:
                structure.update(await self._parse_js_structure(code_content))
                
        except Exception as e:
            structure["parsing_error"] = str(e)
            
        return structure
    
    async def _run_quality_checks(self, code_content: str, code_directory: str, language: str) -> List[Dict[str, Any]]:
        """运行代码质量检查工具"""
        quality_issues = []
        
        if language == "python":
            # Pylint检查
            if self.available_tools.get("pylint"):
                pylint_issues = await self._run_pylint(code_content, code_directory)
                quality_issues.extend(pylint_issues)
            
            # Flake8检查
            if self.available_tools.get("flake8"):
                flake8_issues = await self._run_flake8(code_content, code_directory)
                quality_issues.extend(flake8_issues)
        
        return quality_issues
    
    async def _run_security_scans(self, code_content: str, code_directory: str, language: str) -> List[Dict[str, Any]]:
        """运行安全漏洞扫描"""
        security_issues = []
        
        if language == "python" and self.available_tools.get("bandit"):
            bandit_issues = await self._run_bandit(code_content, code_directory)
            security_issues.extend(bandit_issues)
        elif language in ["cpp", "c"]:
            if self.available_tools.get("semgrep"):
                semgrep_issues = await self._run_semgrep(code_content, code_directory, language)
                security_issues.extend(semgrep_issues)
            if self.available_tools.get("cppcheck"):
                cppcheck_issues = await self._run_cppcheck(code_content, code_directory)
                security_issues.extend(cppcheck_issues)
            if self.available_tools.get("clang-tidy"):
                clang_issues = await self._run_clang_tidy(code_content, code_directory, language)
                security_issues.extend(clang_issues)
        elif language == "java":
            if self.available_tools.get("semgrep"):
                semgrep_issues = await self._run_semgrep(code_content, code_directory, language)
                security_issues.extend(semgrep_issues)
            if self.available_tools.get("spotbugs"):
                spotbugs_issues = await self._run_spotbugs(code_content, code_directory)
                security_issues.extend(spotbugs_issues)
            
        return security_issues

    async def _run_external_tool(self, command: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """执行外部命令并统一处理异常。"""
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    def _write_temp_source(self, code_content: str, suffix: str = ".c") -> str:
        """写入临时代码文件并返回路径。"""
        fd, temp_path = tempfile.mkstemp(prefix="static_scan_", suffix=suffix)
        os.close(fd)
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        return temp_path

    async def _run_semgrep(self, code_content: str, code_directory: str, language: str) -> List[Dict[str, Any]]:
        """运行Semgrep安全扫描。"""
        issues = []
        temp_file = None
        try:
            suffix = ".c" if language in ["cpp", "c"] else ".txt"
            temp_file = self._write_temp_source(code_content, suffix=suffix)
            timeout = self.agent_config.get("semgrep_timeout", 60)
            result = await self._run_external_tool([
                "semgrep", "--config", "auto", "--json", "--quiet", temp_file
            ], timeout=timeout)

            # semgrep发现问题时可能返回非0，此时stderr为空且stdout仍有结果
            if result.stdout:
                data = json.loads(result.stdout)
                for item in data.get("results", []):
                    extra = item.get("extra", {})
                    metadata = extra.get("metadata", {})
                    start = item.get("start", {})
                    issues.append({
                        "tool": "semgrep",
                        "type": "security",
                        "message": extra.get("message", ""),
                        "line": start.get("line", 0),
                        "column": start.get("col", 0),
                        "severity": self._map_semgrep_severity(extra.get("severity", "INFO")),
                        "rule_id": item.get("check_id", ""),
                        "cwe": metadata.get("cwe", "")
                    })
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ Semgrep运行失败: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        return issues

    async def _run_cppcheck(self, code_content: str, code_directory: str) -> List[Dict[str, Any]]:
        """运行Cppcheck并解析XML输出。"""
        issues = []
        temp_file = None
        try:
            temp_file = self._write_temp_source(code_content, suffix=".c")
            timeout = self.agent_config.get("cppcheck_timeout", 60)
            result = await self._run_external_tool([
                "cppcheck", "--enable=all", "--xml", "--xml-version=2", temp_file
            ], timeout=timeout)

            # cppcheck XML通常在stderr输出
            xml_output = result.stderr or ""
            if xml_output.strip():
                issues.extend(self._parse_cppcheck_xml(xml_output))
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ Cppcheck运行失败: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        return issues

    async def _run_clang_tidy(self, code_content: str, code_directory: str, language: str) -> List[Dict[str, Any]]:
        """运行clang-tidy并解析文本输出。"""
        issues = []
        temp_file = None
        try:
            suffix = ".cpp" if language == "cpp" else ".c"
            temp_file = self._write_temp_source(code_content, suffix=suffix)
            timeout = self.agent_config.get("clang_tidy_timeout", 60)
            result = await self._run_external_tool([
                "clang-tidy", temp_file, "--", "-std=c11"
            ], timeout=timeout)
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if output.strip():
                issues.extend(self._parse_clang_tidy_output(output))
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ clang-tidy运行失败: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        return issues

    async def _run_spotbugs(self, code_content: str, code_directory: str) -> List[Dict[str, Any]]:
        """运行SpotBugs并解析XML输出。

        说明: spotbugs通常针对编译后的.class/.jar，这里对单文件源码场景仅做能力接入。
        若code_directory下未检测到class/jar，返回空结果。
        """
        issues = []
        try:
            if not code_directory or not os.path.isdir(code_directory):
                return issues

            has_targets = False
            for root, _, files in os.walk(code_directory):
                for name in files:
                    if name.endswith(".class") or name.endswith(".jar"):
                        has_targets = True
                        break
                if has_targets:
                    break

            if not has_targets:
                return issues

            timeout = self.agent_config.get("spotbugs_timeout", 120)
            result = await self._run_external_tool([
                "spotbugs", "-textui", "-effort:max", "-low", "-xml:withMessages", code_directory
            ], timeout=timeout)
            xml_output = result.stdout or result.stderr or ""
            if xml_output.strip():
                issues.extend(self._parse_spotbugs_xml(xml_output))
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ SpotBugs运行失败: {e}")
        return issues

    def _parse_cppcheck_xml(self, xml_output: str) -> List[Dict[str, Any]]:
        """解析Cppcheck XML并归一化为issue列表。"""
        issues: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_output)
            errors = root.findall(".//error")
            for error in errors:
                loc = error.find("location")
                line = int(loc.get("line", "0")) if loc is not None else 0
                file_path = loc.get("file", "") if loc is not None else ""
                issues.append({
                    "tool": "cppcheck",
                    "type": "security",
                    "message": error.get("msg", ""),
                    "line": line,
                    "file": file_path,
                    "severity": self._map_cppcheck_severity(error.get("severity", "style")),
                    "rule_id": error.get("id", "")
                })
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ Cppcheck XML解析失败: {e}")
        return issues

    def _map_cppcheck_severity(self, severity: str) -> str:
        """将cppcheck严重等级映射到统一等级。"""
        mapping = {
            "error": "high",
            "warning": "medium",
            "style": "low",
            "performance": "low",
            "portability": "low",
            "information": "low"
        }
        return mapping.get((severity or "").lower(), "low")

    def _map_semgrep_severity(self, severity: str) -> str:
        """将semgrep严重等级映射到统一等级。"""
        mapping = {
            "ERROR": "high",
            "WARNING": "medium",
            "INFO": "low"
        }
        return mapping.get((severity or "").upper(), "low")

    def _parse_clang_tidy_output(self, output: str) -> List[Dict[str, Any]]:
        """解析clang-tidy文本输出。"""
        issues: List[Dict[str, Any]] = []
        # 典型格式: file.c:10:5: warning: message [check-name]
        pattern = re.compile(r"^(.*?):(\d+):(\d+):\s+(warning|error|note):\s+(.*?)\s*(\[.*\])?$")
        for line in output.splitlines():
            line = line.strip()
            match = pattern.match(line)
            if not match:
                continue
            file_path, line_no, col_no, level, message, check = match.groups()
            issues.append({
                "tool": "clang-tidy",
                "type": "security" if level in ["warning", "error"] else "style",
                "message": message,
                "line": int(line_no),
                "column": int(col_no),
                "file": file_path,
                "severity": "high" if level == "error" else "medium",
                "rule_id": (check or "").strip("[]")
            })
        return issues

    def _parse_spotbugs_xml(self, xml_output: str) -> List[Dict[str, Any]]:
        """解析SpotBugs XML输出。"""
        issues: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_output)
            for bug in root.findall(".//BugInstance"):
                priority = bug.get("priority", "3")
                bug_type = bug.get("type", "")
                short_msg = bug.findtext("ShortMessage", default="")
                long_msg = bug.findtext("LongMessage", default="")
                src = bug.find(".//SourceLine")
                line = int(src.get("start", "0")) if src is not None and src.get("start") else 0
                source_path = src.get("sourcepath", "") if src is not None else ""
                issues.append({
                    "tool": "spotbugs",
                    "type": "security",
                    "message": long_msg or short_msg,
                    "line": line,
                    "file": source_path,
                    "severity": self._map_spotbugs_priority(priority),
                    "rule_id": bug_type
                })
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ SpotBugs XML解析失败: {e}")
        return issues

    def _map_spotbugs_priority(self, priority: str) -> str:
        """SpotBugs priority(1高,3低)映射。"""
        mapping = {
            "1": "high",
            "2": "medium",
            "3": "low"
        }
        return mapping.get(str(priority), "low")
    
    async def _run_complexity_analysis(self, code_content: str, code_directory: str, language: str) -> Dict[str, Any]:
        """运行复杂度分析"""
        complexity_data = {
            "cyclomatic_complexity": {},
            "maintainability_index": 0.0,
            "halstead_metrics": {},
            "average_complexity": 0.0
        }
        
        if language == "python" and self.available_tools.get("radon"):
            complexity_data = await self._run_radon_analysis(code_content, code_directory)
            
        return complexity_data
    
    async def _run_type_checks(self, code_content: str, code_directory: str, language: str) -> List[Dict[str, Any]]:
        """运行类型检查"""
        type_issues = []
        
        if language == "python" and self.available_tools.get("mypy"):
            mypy_issues = await self._run_mypy(code_content, code_directory)
            type_issues.extend(mypy_issues)
            
        return type_issues
    
    async def _run_style_checks(self, code_content: str, code_directory: str, language: str) -> List[Dict[str, Any]]:
        """运行代码风格检查"""
        style_issues = []
        
        if language == "python":
            # 基础PEP8检查
            pep8_issues = await self._check_pep8_compliance(code_content)
            style_issues.extend(pep8_issues)
            
        return style_issues
    
    async def _run_pylint(self, code_content: str, code_directory: str) -> List[Dict[str, Any]]:
        """运行Pylint分析"""
        issues = []
        
        try:
            # 将代码写入临时文件
            temp_file = "/tmp/code_analysis.py"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            # 运行pylint
            pylint_timeout = self.agent_config.get("pylint_timeout", 60)
            result = subprocess.run([
                "pylint", temp_file, "--output-format=json", "--score=no"
            ], capture_output=True, text=True, timeout=pylint_timeout)
            
            if result.stdout:
                pylint_data = json.loads(result.stdout)
                for item in pylint_data:
                    issues.append({
                        "tool": "pylint",
                        "type": item.get("type", "unknown"),
                        "message": item.get("message", ""),
                        "line": item.get("line", 0),
                        "column": item.get("column", 0),
                        "severity": self._map_pylint_severity(item.get("type")),
                        "symbol": item.get("symbol", ""),
                        "message_id": item.get("message-id", "")
                    })
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ Pylint运行失败: {e}")
            
        return issues
    
    async def _run_flake8(self, code_content: str, code_directory: str) -> List[Dict[str, Any]]:
        """运行Flake8分析"""
        issues = []
        
        try:
            temp_file = "/tmp/code_analysis.py"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            result = subprocess.run([
                "flake8", temp_file, "--format=json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':')
                        if len(parts) >= 4:
                            issues.append({
                                "tool": "flake8",
                                "type": "style",
                                "message": ':'.join(parts[3:]).strip(),
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "column": int(parts[2]) if parts[2].isdigit() else 0,
                                "severity": "low",
                                "code": parts[3].strip().split()[0] if parts[3].strip() else ""
                            })
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ Flake8运行失败: {e}")
            
        return issues
    
    async def _run_bandit(self, code_content: str, code_directory: str) -> List[Dict[str, Any]]:
        """运行Bandit安全扫描"""
        issues = []
        
        try:
            temp_file = "/tmp/code_analysis.py"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            result = subprocess.run([
                "bandit", "-f", "json", temp_file
            ], capture_output=True, text=True, timeout=30)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                for item in bandit_data.get("results", []):
                    issues.append({
                        "tool": "bandit",
                        "type": "security",
                        "message": item.get("issue_text", ""),
                        "line": item.get("line_number", 0),
                        "severity": item.get("issue_severity", "low").lower(),
                        "confidence": item.get("issue_confidence", "low").lower(),
                        "test_id": item.get("test_id", ""),
                        "test_name": item.get("test_name", "")
                    })
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ Bandit运行失败: {e}")
            
        return issues
    
    async def _run_radon_analysis(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """运行Radon复杂度分析"""
        complexity_data = {
            "cyclomatic_complexity": {},
            "maintainability_index": 0.0,
            "average_complexity": 0.0
        }
        
        try:
            temp_file = "/tmp/code_analysis.py"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            # 圈复杂度分析
            cc_result = subprocess.run([
                "radon", "cc", temp_file, "-j"
            ], capture_output=True, text=True, timeout=30)
            
            if cc_result.stdout:
                cc_data = json.loads(cc_result.stdout)
                complexity_data["cyclomatic_complexity"] = cc_data
            
            # 可维护性指数
            mi_result = subprocess.run([
                "radon", "mi", temp_file, "-j"
            ], capture_output=True, text=True, timeout=30)
            
            if mi_result.stdout:
                mi_data = json.loads(mi_result.stdout)
                complexity_data["maintainability_index"] = mi_data
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ Radon分析失败: {e}")
            
        return complexity_data
    
    async def _run_mypy(self, code_content: str, code_directory: str) -> List[Dict[str, Any]]:
        """运行MyPy类型检查"""
        issues = []
        
        try:
            temp_file = "/tmp/code_analysis.py"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            result = subprocess.run([
                "mypy", temp_file, "--no-error-summary"
            ], capture_output=True, text=True, timeout=30)
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 3:
                            issues.append({
                                "tool": "mypy",
                                "type": "type_error",
                                "message": ':'.join(parts[2:]).strip(),
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "severity": "medium"
                            })
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            log("static_scan_tools", LogLevel.WARNING, f"⚠️ MyPy运行失败: {e}")
            
        return issues

    async def _check_pep8_compliance(self, code_content: str) -> List[Dict[str, Any]]:
        """检查PEP8编码规范遵循情况"""
        issues = []
        
        lines = code_content.split('\n')
        for i, line in enumerate(lines, 1):
            # 行长度检查
            if len(line) > 79:
                issues.append({
                    "tool": "pep8_checker",
                    "type": "style",
                    "message": f"Line too long ({len(line)} > 79 characters)",
                    "line": i,
                    "severity": "low",
                    "code": "E501"
                })
            
            # 缩进检查(简化版)
            if line.startswith(' ') and not line.startswith('    '):
                stripped = line.lstrip()
                if stripped and not line.startswith('    '):
                    issues.append({
                        "tool": "pep8_checker",
                        "type": "style", 
                        "message": "Indentation is not a multiple of four",
                        "line": i,
                        "severity": "low",
                        "code": "E111"
                    })
        
        return issues
    
    async def _generate_scan_summary(self, quality_issues, security_issues, complexity_analysis, 
                                   type_issues, style_issues) -> Dict[str, Any]:
        """生成扫描结果摘要"""
        total_issues = len(quality_issues) + len(security_issues) + len(type_issues) + len(style_issues)
        
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        all_issues = quality_issues + security_issues + type_issues + style_issues
        for issue in all_issues:
            severity = issue.get("severity", "low")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # 计算质量分数
        quality_score = max(0.0, 10.0 - (
            severity_counts["critical"] * 3.0 +
            severity_counts["high"] * 2.0 +
            severity_counts["medium"] * 1.0 +
            severity_counts["low"] * 0.5
        ))
        
        return {
            "total_issues": total_issues,
            "severity_breakdown": severity_counts,
            "quality_score": round(quality_score, 1),
            "quality_grade": self._score_to_grade(quality_score),
            "has_security_issues": len(security_issues) > 0,
            "has_type_issues": len(type_issues) > 0,
            "maintainability_index": complexity_analysis.get("maintainability_index", 0.0),
            "recommendations": self._generate_recommendations(severity_counts, total_issues)
        }
    
    def _map_pylint_severity(self, pylint_type: str) -> str:
        """映射Pylint消息类型到严重程度"""
        severity_map = {
            "error": "high",
            "warning": "medium", 
            "refactor": "low",
            "convention": "low",
            "info": "low"
        }
        return severity_map.get(pylint_type, "low")
    
    def _score_to_grade(self, score: float) -> str:
        """将分数转换为等级"""
        if score >= 9.0:
            return "A"
        elif score >= 8.0:
            return "B"
        elif score >= 7.0:
            return "C"
        elif score >= 6.0:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, severity_counts: Dict[str, int], total_issues: int) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if severity_counts["critical"] > 0:
            recommendations.append("立即修复严重问题,代码可能存在安全风险或逻辑错误")
        
        if severity_counts["high"] > 0:
            recommendations.append("优先修复高级别问题,这些问题影响代码质量和可维护性")
        
        if total_issues > 50:
            recommendations.append("问题数量过多,建议分模块逐步改进")
        
        if severity_counts["low"] > 20:
            recommendations.append("考虑配置代码格式化工具自动修复样式问题")
        
        return recommendations
    
    def _detect_language(self, code_content: str) -> str:
        """检测编程语言"""
        if "def " in code_content and "import " in code_content:
            return "python"
        elif "function " in code_content or "const " in code_content:
            return "javascript"
        elif "class " in code_content and "public " in code_content:
            return "java"
        elif "#include" in code_content:
            return "c" if "std::" not in code_content else "cpp"
        elif re.search(r"\b(int|char|void|size_t|struct)\s+\w+\s*\(", code_content):
            return "cpp"
        else:
            return "unknown"

    async def _parse_python_structure(self, code_content: str) -> Dict[str, Any]:
        """解析Python代码结构"""
        structure = {
            "functions": [],
            "classes": [],
            "imports": [],
            "constants": []
        }
        
        try:
            lines = code_content.split('\n')
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                if stripped.startswith('def '):
                    func_name = stripped.split('def ')[1].split('(')[0]
                    structure["functions"].append({
                        "name": func_name,
                        "line_number": i + 1,
                        "complexity": self._estimate_function_complexity(lines, i)
                    })
                
                elif stripped.startswith('class '):
                    class_name = stripped.split('class ')[1].split('(')[0].split(':')[0]
                    structure["classes"].append({
                        "name": class_name,
                        "line_number": i + 1
                    })
                
                elif stripped.startswith('import ') or stripped.startswith('from '):
                    structure["imports"].append(stripped)
                
                elif '=' in stripped and stripped.split('=')[0].strip().isupper():
                    var_name = stripped.split('=')[0].strip()
                    if var_name.isupper():
                        structure["constants"].append(var_name)
        
        except Exception as e:
            structure["parsing_error"] = str(e)
        
        return structure

    async def _parse_js_structure(self, code_content: str) -> Dict[str, Any]:
        """解析JavaScript代码结构"""
        structure = {
            "functions": [],
            "classes": [],
            "imports": [],
            "constants": []
        }
        
        lines = code_content.split('\n')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if 'function ' in stripped:
                func_match = re.search(r'function\s+(\w+)', stripped)
                if func_match:
                    structure["functions"].append({
                        "name": func_match.group(1),
                        "line_number": i + 1
                    })
            
            elif stripped.startswith('class '):
                class_match = re.search(r'class\s+(\w+)', stripped)
                if class_match:
                    structure["classes"].append({
                        "name": class_match.group(1),
                        "line_number": i + 1
                    })
            
            elif 'import ' in stripped or 'require(' in stripped:
                structure["imports"].append(stripped)
        
        return structure

    def _estimate_function_complexity(self, lines: List[str], start_line: int) -> int:
        """估算函数复杂度"""
        complexity = 1  # 基础复杂度
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, min(start_line + 50, len(lines))):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                break
            
            # 复杂度增加因子
            if any(keyword in line for keyword in ['if ', 'elif ', 'for ', 'while ', 'try:', 'except:']):
                complexity += 1
        
        return complexity

    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行传统静态代码分析任务"""
        return await self._traditional_static_analysis(
            task_data.get("code_content", ""),
            task_data.get("code_directory", "")
        )
