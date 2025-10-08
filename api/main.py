import click
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json
import re

# 配置日志，只在错误时显示详细信息
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from infrastructure.reports import report_manager
except ImportError:
    report_manager = None

@click.group()
def mas():
    """MultiAgentSystem (MAS) - AI代码审查助手"""
    pass

async def _init_system():
    from core.agents_integration import get_agent_integration_system
    agent_system = get_agent_integration_system()
    if not agent_system._system_ready:
        await agent_system.initialize_system()
    return agent_system

async def _dispatch_directory_analysis(agent_system, target_dir: str):
    return await agent_system.analyze_directory(target_dir)

async def _async_wait_for_reports(run_id: str, total_files: int, timeout: int = 1200, poll_interval: int = 10):
    """异步等待分析结果，实时输出进度。"""
    analysis_dir = Path(__file__).parent.parent / 'reports' / 'analysis'
    start = asyncio.get_event_loop().time()
    sum_pat = re.compile(rf"run_summary_.*_{re.escape(run_id)}\.json$")
    cons_pat = re.compile(rf"consolidated_req_\d+_{re.escape(run_id)}_.*\.json$")
    printed_cycles = -1
    summary_file = None
    severity_agg = {"critical":0,"high":0,"medium":0,"low":0,"info":0}
    last_consolidated_count = 0

    def _scan():
        nonlocal summary_file
        consolidated = []
        if analysis_dir.exists():
            for f in analysis_dir.iterdir():
                n = f.name
                if summary_file is None and sum_pat.match(n):
                    summary_file = f
                elif cons_pat.match(n):
                    consolidated.append(f)
        return consolidated

    click.echo(f"⏳ 正在等待分析结果 (最长 {timeout}s，每 {poll_interval}s 刷新)...")
    while True:
        elapsed = int(asyncio.get_event_loop().time() - start)
        if elapsed >= timeout:
            click.echo("⏱️ 超时: 仍未生成运行级汇总。稍后可使用 'mas results <run_id>' 查询。")
            return
        consolidated = _scan()
        # 统计
        severity_agg = {"critical":0,"high":0,"medium":0,"low":0,"info":0}
        total_issues = 0
        for f in consolidated:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                sev = data.get('severity_stats', {})
                for k,v in sev.items():
                    if k in severity_agg:
                        severity_agg[k] += v
                total_issues += data.get('issue_count',0)
            except Exception:
                continue
        cycle = elapsed // poll_interval
        if cycle != printed_cycles:
            printed_cycles = cycle
            if len(consolidated) != last_consolidated_count or cycle % 3 == 0:  # 降低刷屏
                last_consolidated_count = len(consolidated)
                click.echo(f"⌛ {elapsed:>4}s | 文件级报告 {len(consolidated)}/{total_files} | 问题:{total_issues} | 严重度:{severity_agg}")
        if summary_file:
            try:
                summary_data = json.loads(summary_file.read_text(encoding='utf-8'))
            except Exception:
                summary_data = {}
            click.echo("\n✅ 分析完成")
            click.echo(f"运行级汇总报告: {summary_file.name}")
            click.echo(f"问题统计: {summary_data.get('severity_stats', {})}")
            click.echo(f"文件级报告数: {len(consolidated)}")
            click.echo(f"使用命令: mas results {run_id} 查看详情")
            return
        await asyncio.sleep(poll_interval)

async def _run_single_analysis_flow(target_dir: str):
    agent_system = await _init_system()
    # 直接派发
    dispatch = await _dispatch_directory_analysis(agent_system, target_dir)
    if dispatch.get('status') != 'dispatched':
        click.echo(f"❌ 派发失败: {dispatch}")
        return
    run_id = dispatch['run_id']
    click.echo(f"🆔 Run ID: {run_id}")
    click.echo(f"📊 已派发 {dispatch.get('total_files')} 个文件, dispatch报告: {dispatch.get('report_path')}")
    await _async_wait_for_reports(run_id, dispatch.get('total_files'))

# ============ 新增异步实现 -> 同步包装 ============
async def _login_entry(target_dir):
    click.echo("\n=====================================")
    click.echo("      MultiAgentSystem (MAS)")
    click.echo("      AI Code Review Assistant")
    click.echo("=====================================")
    click.echo(f"Login successful at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if target_dir:
        if not os.path.isdir(target_dir):
            click.echo(f"❌ 目录不存在: {target_dir}")
            return
        target_dir = str(Path(target_dir).resolve())
        click.echo(f"📂 目标目录: {target_dir}")
        await _run_single_analysis_flow(target_dir)
    else:
        click.echo("未提供 --target-dir，仅初始化系统供后续交互。")
        await _init_system()
        click.echo("系统初始化完成。可使用 'mas login -d <dir>' 直接分析。")

@mas.command()
@click.option('--target-dir', '-d', help='Directory containing code to review')
def login(target_dir):
    """Login 并可选启动目录分析 (同步包装)"""
    asyncio.run(_login_entry(target_dir))

async def _status_entry():
    click.echo("\n🔍 检查AI智能体系统状态...")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from core.agents_integration import get_agent_integration_system
        from core.ai_agent_config import get_ai_agent_config
        agent_system = await _init_system()
        config_manager = get_ai_agent_config()
        click.echo("\n📋 配置状态:")
        summary = config_manager.get_config_summary()
        click.echo(f"运行模式: {summary['agent_mode']}")
        click.echo(f"配置有效: {'✅' if summary['config_valid'] else '❌'}")
        status = await agent_system.get_agent_status()
        click.echo(f"\n🤖 系统状态: {'✅ 就绪' if status['system_ready'] else '❌ 未就绪'}")
        click.echo("\n📋 智能体列表:")
        active_agents = agent_system.get_active_agents()
        for name, class_name in active_agents.items():
            ai_indicator = "🤖" if name.startswith('ai_') else "🔧"
            click.echo(f"  {ai_indicator} {name}: {class_name}")
        if report_manager:
            click.echo("\n📊 报告系统状态:")
            reports = report_manager.list_reports()
            total_reports = sum(len(files) for files in reports.values())
            click.echo(f"  📄 总报告数: {total_reports}")
            for report_type, files in reports.items():
                if files:
                    click.echo(f"  📁 {report_type}: {len(files)} 个")
        else:
            click.echo("\n⚠️ 报告管理系统不可用")
    except Exception as e:
        click.echo(f"❌ 状态检查失败: {e}")
        import traceback
        logger.error(f"状态检查错误: {traceback.format_exc()}")

@mas.command()
def status():
    """系统状态 (同步包装)"""
    asyncio.run(_status_entry())

# ============ 其余命令保持不变 (results / config) ============
@mas.command()
def config():
    """Configure AI agent system settings"""
    click.echo("\n🤖 AI智能体系统配置")
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from core.ai_agent_config import get_ai_agent_config, print_config_status
        
        config_manager = get_ai_agent_config()
        
        while True:
            click.echo("\n" + "="*50)
            print_config_status()
            click.echo("\n配置选项:")
            click.echo("1. 显示详细配置")
            click.echo("2. 测试AI模型连接")
            click.echo("3. 重置为默认配置")
            click.echo("0. 退出配置")
            
            choice = input("\n请选择 (0-3): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                click.echo("\n📋 详细配置信息:")
                summary = config_manager.get_config_summary()
                click.echo(f"配置有效性: {'✅' if summary['config_valid'] else '❌'}")
                click.echo(f"AI模型超时: {config_manager.get_ai_model_timeout()}秒")
                click.echo(f"最大代码长度: {config_manager.get_max_code_length()}字符")
                click.echo(f"AI置信度阈值: {config_manager.get_ai_confidence_threshold()}")
                click.echo(f"最大并发AI任务: {config_manager.get_max_concurrent_ai_tasks()}")
            elif choice == '2':
                click.echo("\n🔍 测试AI模型连接... (请在运行中的分析流程中进行)")
            elif choice == '3':
                confirm = input("⚠️ 确定要重置为默认配置吗? (y/N): ").strip().lower()
                if confirm == 'y':
                    config_manager.reset_to_defaults()
                    click.echo("✅ 配置已重置为默认值")
                else:
                    click.echo("❌ 取消重置")
            else:
                click.echo("❌ 无效选择，请重试")
                
    except ImportError as e:
        click.echo(f"❌ 无法加载配置系统: {e}")
    except Exception as e:
        click.echo(f"❌ 配置过程中出错: {e}")

@mas.command()
def reports():
    """管理分析报告"""
    if not report_manager:
        click.echo("❌ 报告管理系统不可用")
        return
    
    click.echo("\n📊 MAS 报告管理系统")
    click.echo("=" * 40)
    
    while True:
        click.echo("\n报告管理选项:")
        click.echo("1. 查看所有报告")
        click.echo("2. 查看特定类型报告")
        click.echo("3. 生成系统状态报告")
        click.echo("4. 清理旧报告")
        click.echo("0. 返回主菜单")
        
        choice = input("\n请选择 (0-4): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            # 显示所有报告
            reports = report_manager.list_reports()
            total_reports = sum(len(files) for files in reports.values())
            
            if total_reports == 0:
                click.echo("\n📋 暂无报告文件")
            else:
                click.echo(f"\n📋 共发现 {total_reports} 个报告文件:")
                for report_type, files in reports.items():
                    if files:
                        click.echo(f"\n📁 {report_type.upper()}:")
                        for file_path in sorted(files):
                            click.echo(f"  - {file_path.name}")
        
        elif choice == '2':
            # 查看特定类型报告
            click.echo("\n选择报告类型:")
            click.echo("1. 分析报告 (analysis)")
            click.echo("2. 兼容性报告 (compatibility)")
            click.echo("3. 部署报告 (deployment)")
            click.echo("4. 测试报告 (testing)")
            
            type_choice = input("请选择 (1-4): ").strip()
            type_map = {'1': 'analysis', '2': 'compatibility', '3': 'deployment', '4': 'testing'}
            
            if type_choice in type_map:
                report_type = type_map[type_choice]
                reports = report_manager.list_reports(report_type)
                files = reports.get(report_type, [])
                
                if files:
                    click.echo(f"\n📁 {report_type.upper()} 报告:")
                    for file_path in sorted(files):
                        click.echo(f"  - {file_path.name}")
                else:
                    click.echo(f"\n📋 暂无 {report_type} 类型的报告")
        
        elif choice == '3':
            # 生成系统状态报告
            click.echo("\n🔍 生成系统状态报告...")
            
            status_data = {
                "报告生成时间": datetime.now().isoformat(),
                "系统版本": "MAS v2.0.0",
                "报告管理器": "✅ 可用" if report_manager else "❌ 不可用",
                "AI模型": "Qwen1.5-7B-Chat",
                "系统状态": "运行正常"
            }
            
            # 获取报告统计
            reports = report_manager.list_reports()
            status_data["报告统计"] = {
                report_type: len(files) for report_type, files in reports.items()
            }
            
            report_path = report_manager.generate_deployment_report(
                f"""# MAS 系统状态报告

## 系统信息
- **生成时间**: {status_data['报告生成时间']}
- **系统版本**: {status_data['系统版本']}
- **AI模型**: {status_data['AI模型']}

## 组件状态
- **报告管理器**: {status_data['报告管理器']}
- **系统状态**: {status_data['系统状态']}

## 报告统计
""" + "\n".join([f"- **{k}**: {v} 个" for k, v in status_data['报告统计'].items()]),
                "system_status_report.md"
            )
            
            click.echo(f"✅ 系统状态报告已生成: {report_path.name}")
        
        elif choice == '4':
            # 清理旧报告
            days = input("请输入保留天数 (默认30天): ").strip()
            try:
                days = int(days) if days else 30
                report_manager.cleanup_old_reports(days)
            except ValueError:
                click.echo("❌ 无效的天数输入")
        
        else:
            click.echo("❌ 无效选择，请重试")

@mas.command()
@click.argument('run_id')
@click.option('--top', default=20, help='Top N issues by severity to display')
def results(run_id, top):
    """显示指定 RUN_ID 的汇总与高严重度问题 (支持新目录结构)"""
    analysis_root = Path(__file__).parent.parent / 'reports' / 'analysis'
    run_dir = analysis_root / run_id
    legacy_mode = False
    if not analysis_root.exists():
        click.echo("❌ 报告目录不存在")
        return
    summary_file = None
    consolidated_files = []
    agent_reports = {}
    if run_dir.exists():
        # 新结构
        summary_file_path = run_dir / 'run_summary.json'
        if summary_file_path.exists():
            summary_file = summary_file_path
        cons_dir = run_dir / 'consolidated'
        if cons_dir.exists():
            consolidated_files = sorted(cons_dir.glob('consolidated_req_*.json'))
        agents_dir = run_dir / 'agents'
        if agents_dir.exists():
            for agent_sub in agents_dir.iterdir():
                if agent_sub.is_dir():
                    agent_reports[agent_sub.name] = sorted(agent_sub.glob('*.json'))
    else:
        # 兼容旧结构
        legacy_mode = True
        sum_pat = re.compile(rf"run_summary_.*_{re.escape(run_id)}\.json$")
        cons_pat = re.compile(rf"consolidated_req_\\d+_{re.escape(run_id)}_.*\.json$")
        for f in analysis_root.iterdir():
            n = f.name
            if sum_pat.match(n):
                summary_file = f
            elif cons_pat.match(n):
                consolidated_files.append(f)
    if not summary_file and not consolidated_files:
        click.echo("⚠️ 未找到对应run的报告文件 (可能仍在分析)")
        return
    severity_order = {"critical":0, "high":1, "medium":2, "low":3, "info":4}
    def load_json(p):
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}
    summary_data = load_json(summary_file) if summary_file else {}
    issues = []
    for f in consolidated_files:
        data = load_json(f)
        for it in data.get('issues', []):
            issues.append(it)
    issues_sorted = sorted(issues, key=lambda x: severity_order.get(x.get('severity','low'), 5))
    click.echo(f"\n📄 Run ID: {run_id}{' (legacy)' if legacy_mode else ''}")
    if summary_file:
        click.echo(f"运行级汇总: {summary_file.relative_to(analysis_root)}")
        sev = summary_data.get('severity_stats', {})
        click.echo(f"问题统计: {sev}")
    click.echo(f"文件级综合报告数量: {len(consolidated_files)}")
    if agent_reports:
        click.echo("\n🧩 Agent单独报告统计:")
        for agent_name, files in agent_reports.items():
            click.echo(f"  - {agent_name}: {len(files)} 个")
    click.echo(f"\n显示前 {min(top, len(issues_sorted))} 条高优先级问题:")
    for i, it in enumerate(issues_sorted[:top], 1):
        click.echo(f"{i}. [{it.get('severity')}] {it.get('file','?')} -> {it.get('description','')} ({it.get('source')})")

if __name__ == '__main__':
    # 对支持原生协程命令的 click>=8.1 会自动运行事件循环
    mas()

main = mas