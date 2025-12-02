import click
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json
import re
import functools

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
    """异步等待分析结果，实时输出进度。
    新结构: reports/analysis/<run_id>/run_summary.json 与 consolidated/consolidated_req_<rid>.json
    旧结构: 顶层 run_summary_*_<run_id>.json 与 consolidated_req_<rid>_<run_id>_*.json
    """
    analysis_dir = Path(__file__).parent.parent / 'reports' / 'analysis'
    start = asyncio.get_event_loop().time()

    # 预编译旧结构正则
    import re as _re
    sum_pat = _re.compile(rf"run_summary_.*_{_re.escape(run_id)}\.json$")
    cons_pat = _re.compile(rf"consolidated_req_\d+_{_re.escape(run_id)}_.*\.json$")

    click.echo(f"⏳ 正在等待分析结果 (最长 {timeout}s，每 {poll_interval}s 刷新)...")

    def _scan_new_structure():
        run_dir = analysis_dir / run_id
        if not run_dir.exists():
            return None
        summary_file = run_dir / 'run_summary.json'
        cons_dir = run_dir / 'consolidated'
        consolidated = []
        if cons_dir.exists():
            consolidated = sorted(cons_dir.glob('consolidated_req_*.json'))
        return {
            'summary': summary_file if summary_file.exists() else None,
            'consolidated': consolidated
        }

    def _scan_old_structure():
        summary_file = None
        consolidated = []
        if analysis_dir.exists():
            for f in analysis_dir.iterdir():
                n = f.name
                if summary_file is None and sum_pat.match(n):
                    summary_file = f
                elif cons_pat.match(n):
                    consolidated.append(f)
        return {'summary': summary_file, 'consolidated': consolidated}

    while True:
        elapsed = int(asyncio.get_event_loop().time() - start)
        if elapsed >= timeout:
            click.echo("⏱️ 超时: 仍未生成运行级汇总。稍后可使用 'mas results <run_id>' 查询。")
            return

        scan_result = _scan_new_structure() or _scan_old_structure()
        summary_file = scan_result['summary']
        consolidated_files = scan_result['consolidated']

        if summary_file:
            try:
                summary_data = json.loads(summary_file.read_text(encoding='utf-8'))
            except Exception:
                summary_data = {}
            click.echo("\n✅ 分析完成")
            if summary_file.parent.name == run_id or summary_file.name == 'run_summary.json':
                rel = summary_file.relative_to(analysis_dir)
            else:
                rel = summary_file.name
            click.echo(f"运行级汇总报告: {rel}")
            sev = summary_data.get('severity_stats') or summary_data.get('summary', {}).get('severity_breakdown', {})
            click.echo(f"问题统计: {sev}")
            click.echo(f"文件级报告数: {len(consolidated_files)}")
            click.echo(f"使用命令: mas results {run_id} 查看详情")
            # 新增：显眼的结束提示
            click.echo("\n🎯 本次分析流程全部结束 ✅")
            click.echo(f"🆔 Run ID: {run_id}")
            click.echo("👉 现在可以继续输入指令、执行 /analyze 新目录或使用 /exit 退出。")
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
    # 新增：单次分析流程结束提示（防止用户等待中断后无反馈）
    click.echo("\n🚀 目录分析任务已完整结束")
    click.echo(f"🧾 可使用: mas results {run_id} 查看详情或在交互模式再次 /analyze 其他目录。")
    click.echo("—— 分析结束 ——")

# ============ 新增异步实现 -> 同步包装 ============
async def _interactive_chat(agent_system):
    """命令行异步交互循环，保持会话不退出。
    支持指令:
      /exit /quit  退出会话
      /analyze <path>  触发目录分析并等待完成，结束时提示
    其他输入将发送给 AI 对话代理。
    """
    print("\n💬 进入交互模式。输入 /exit 或 /quit 退出，会话保持。")
    print("📌 支持指令: /analyze <目录路径> | /exit")
    while True:
        try:
            user = await asyncio.to_thread(lambda: input("你> ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\n👋 检测到退出信号，结束会话。")
            break
        if not user:
            continue
        low = user.lower()
        if low in {"/exit", "/quit"}:
            print("👋 会话结束，感谢使用 MAS。")
            break
        if user.startswith("/analyze "):
            target_dir = user[len("/analyze "):].strip()
            if not target_dir:
                print("❌ 未提供目录路径")
                continue
            print(f"🚀 触发目录分析: {target_dir}")
            try:
                result = await agent_system.analyze_directory(target_dir)
                status = result.get('status')
                if status == 'dispatched':
                    run_id = result.get('run_id')
                    total_files = result.get('total_files')
                    print(f"✅ 已派发 {total_files} 个文件，run_id={run_id}")
                    print("⏳ 等待综合报告生成，过程可能较长，期间仍会输出进度...")
                    # 调用主文件的进度等待函数
                    await _async_wait_for_reports(run_id, total_files)
                    # 明确结束通知
                    print(f"🎯 分析结束 run_id={run_id} ✅ 输入: results {run_id} 查看详情 或继续输入新的指令。")
                elif status == 'empty':
                    print("⚠️ 目录中未找到可分析的Python文件")
                else:
                    print(f"❌ 分析失败: {result.get('message','未知错误')}")
            except Exception as e:
                print(f"❌ 分析异常: {e}")
            continue
        # 普通对话消息
        resp = await agent_system.process_message_from_cli(user)
        if not resp.startswith("✅"):
            print(resp)

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
        # 可选：分析后进入交互
        click.echo("📥 分析流程结束，进入交互会话。输入 /exit 退出。")
        agent_system = await _init_system()
        await _interactive_chat(agent_system)
    else:
        click.echo("未提供 --target-dir，仅初始化系统供后续交互。")
        agent_system = await _init_system()
        click.echo("系统初始化完成。可使用 '/analyze <dir>' 或命令行 'mas login -d <dir>' 分析。")
        # 进入持久交互循环
        await _interactive_chat(agent_system)

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
        from infrastructure.config.ai_agents import get_ai_agent_config
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
        from infrastructure.config.ai_agents import get_ai_agent_config, print_config_status
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
                click.echo("👋 退出配置菜单")
                break
            elif choice == '1':
                click.echo("\n📋 详细配置信息:")
                summary = config_manager.get_config_summary()
                click.echo(f"配置有效性: {'✅' if summary['config_valid'] else '❌'}")
                click.echo(f"运行模式: {summary['agent_mode']}")
                click.echo(f"AI模型超时: {config_manager.get_ai_model_timeout()}秒")
                click.echo(f"最大代码长度: {config_manager.get_max_code_length()}字符")
                click.echo(f"AI置信度阈值: {config_manager.get_ai_confidence_threshold()}")
                click.echo(f"最大并发AI任务: {config_manager.get_max_concurrent_ai_tasks()}")
            elif choice == '2':
                click.echo("\n🔍 测试AI模型连接...")
                # 简单调用: 初始化系统并测试可用 AI agent
                try:
                    agent_system = asyncio.run(_init_system())  # 在同步上下文里直接运行新loop
                    test = asyncio.run(agent_system.test_ai_agents())
                    for name, result in test.items():
                        status = result.get('status')
                        ai_ready = result.get('ai_ready')
                        click.echo(f"  {name}: {status} | AI就绪: {ai_ready}")
                except Exception as e:
                    click.echo(f"❌ 测试失败: {e}")
            elif choice == '3':
                confirm = input("⚠️ 确认重置为默认配置? (y/N): ").strip().lower()
                if confirm == 'y':
                    config_manager.reset_to_defaults()
                    click.echo("✅ 已重置为默认配置")
                else:
                    click.echo("❌ 取消重置")
            else:
                click.echo("❌ 无效选择，请重试")
    except ImportError as e:
        click.echo(f"❌ 无法加载配置系统: {e}")
    except Exception as e:
        click.echo(f"❌ 配置过程中出错: {e}")

# 重新添加 results 命令（如之前被截断）
@mas.command()
@click.argument('run_id')
@click.option('--top', default=20, help='Top N issues by severity to display')
def results(run_id, top):
    """显示指定 RUN_ID 的汇总与高严重度问题"""
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
        click.echo("⚠️ 未找到对应run的报告文件")
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
    mas()

main = mas