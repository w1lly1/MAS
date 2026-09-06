import click
import os
import sys
import asyncio
import logging
import signal
import atexit
from pathlib import Path
from datetime import datetime
import json
import re
import functools

# Determine project root early so we can locate a .env file there
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env from project root if present. Prefer python-dotenv when available,
# otherwise fall back to a simple parser that sets environment variables.
from pathlib import Path as _Path_for_env
_env_path = _Path_for_env(project_root) / '.env'
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_env_path)
except Exception:
    if _env_path.exists():
        for _line in _env_path.read_text(encoding='utf-8').splitlines():
            _line = _line.strip()
            if not _line or _line.startswith('#'):
                continue
            if '=' in _line:
                _k, _v = _line.split('=', 1)
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v

# keep transformer env tweaks
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# 缓解 CUDA 显存碎片化导致的 OOM（Qwen+codebert+distilbert 同驻 24G 时贴边）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

try:
    from infrastructure.reports import report_manager
except ImportError:
    report_manager = None

@click.group()
def mas():
    """Multi-Agent System (MAS) - AI代码审查助手"""
    pass

def _disconnect_agent_vector_services(agent_system) -> None:
    agents = getattr(agent_system, "agents", {}) or {}
    for agent in agents.values():
        vector_service = getattr(agent, "vector_service", None)
        if vector_service is None:
            continue
        try:
            vector_service.disconnect()
        except Exception:
            pass


# Graceful shutdown helper: ensure Weaviate client disconnects on signals/exit
def _graceful_shutdown(signum=None, frame=None):
    try:
        from core.agents_integration import get_agent_integration_system
        agent_system = get_agent_integration_system()
        # Try immediate synchronous disconnect of all Weaviate clients if available
        _disconnect_agent_vector_services(agent_system)
        # Attempt async shutdown of agents (best-effort)
        try:
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except Exception:
                loop = None

            if loop is not None and loop.is_running():
                try:
                    loop.call_soon_threadsafe(lambda: asyncio.create_task(agent_system.shutdown_system()))
                except Exception:
                    # fallback: run in a separate thread
                    import threading
                    threading.Thread(target=lambda: asyncio.run(agent_system.shutdown_system()), daemon=True).start()
            else:
                try:
                    asyncio.run(agent_system.shutdown_system())
                except Exception:
                    pass
        except Exception:
            pass
    finally:
        # If called from a signal handler (signum is not None), force exit immediately.
        # Do NOT call sys.exit() when this function is invoked by atexit (signum is None),
        # because raising SystemExit inside an atexit callback can produce warnings.
        try:
            if signum is not None:
                os._exit(0)
        except Exception:
            pass


# register signal handlers and atexit
try:
    signal.signal(signal.SIGINT, _graceful_shutdown)
except Exception:
    pass
try:
    signal.signal(signal.SIGTERM, _graceful_shutdown)
except Exception:
    pass
try:
    atexit.register(_graceful_shutdown)
except Exception:
    pass

_use_cpu_mode = False

async def _init_system():
    from core.agents_integration import get_agent_integration_system
    agent_system = get_agent_integration_system()
    if not agent_system._system_ready:
        await agent_system.initialize_system(use_cpu_mode=_use_cpu_mode)
    return agent_system

async def _dispatch_directory_analysis(agent_system, target_dir: str, output_dir: str | None = None):
    return await agent_system.analyze_directory(target_dir, output_dir=output_dir)

async def _async_wait_for_reports(agent_system, run_id: str, total_files: int, timeout: int | None = None, poll_interval: int = 10):
    """异步等待分析结果进入稳定状态，并打印最终摘要。"""
    if timeout is None:
        ai_config = getattr(agent_system, "ai_config", None)
        if ai_config and hasattr(ai_config, "get_user_communication_agent_config"):
            timeout = ai_config.get_user_communication_agent_config().get("analysis_wait_timeout")
    if timeout is None:
        timeout = 1800
    click.echo(f"⏳ 正在等待分析结果 (最长 {timeout}s，每 {poll_interval}s 刷新)...")

    try:
        completion = await agent_system.wait_for_run_completion(
            run_id,
            timeout=float(timeout),
            poll_interval=float(poll_interval),
            quiet_period=30.0,
        )
    except Exception as e:
        click.echo(f"❌ 等待分析结果失败: {e}")
        return

    if completion.get('status') != 'completed':
        click.echo("⏱️ 超时: 仍未生成稳定的运行级汇总。稍后可使用 'mas results <run_id>' 查询。")
        return

    summary_path = completion.get('summary_report')
    consolidated_files = completion.get('consolidated_reports', [])
    summary_data = {}
    if summary_path:
        try:
            summary_data = json.loads(Path(summary_path).read_text(encoding='utf-8'))
        except Exception:
            summary_data = {}

    click.echo("\n✅ 分析完成")
    if summary_path:
        try:
            summary_rel = Path(summary_path)
            analysis_dir = Path(__file__).parent.parent / 'reports' / 'analysis'
            if analysis_dir in summary_rel.parents:
                rel = summary_rel.relative_to(analysis_dir)
            else:
                rel = summary_rel.name
        except Exception:
            rel = summary_path
        click.echo(f"运行级汇总报告: {rel}")
    sev = summary_data.get('severity_stats') or summary_data.get('summary', {}).get('severity_breakdown', {})
    click.echo(f"问题统计: {sev}")
    click.echo(f"文件级报告数: {len(consolidated_files)}")
    click.echo(f"使用命令: mas results {run_id} 查看详情")
    click.echo("\n🎯 本次分析流程全部结束 ✅")
    click.echo(f"🆔 Run ID: {run_id}")
    click.echo("👉 现在可以继续输入指令、执行 /analyze 新目录或使用 /exit 退出。")

async def _run_single_analysis_flow(target_dir: str, output_dir: str | None = None):
    agent_system = await _init_system()
    # 直接派发
    dispatch = await _dispatch_directory_analysis(agent_system, target_dir, output_dir=output_dir)
    if dispatch.get('status') != 'dispatched':
        click.echo(f"❌ 派发失败: {dispatch}")
        return
    run_id = dispatch['run_id']
    report_rel = dispatch.get('report_relpath') or run_id
    click.echo(f"🆔 Run ID: {run_id}")
    click.echo(f"📁 报告目录: reports/analysis/{report_rel}")
    click.echo(f"📊 已派发 {dispatch.get('total_files')} 个文件, dispatch报告: {dispatch.get('report_path')}")
    await _async_wait_for_reports(
        agent_system,
        run_id,
        dispatch.get('total_files'),
        timeout=dispatch.get('estimated_timeout_seconds'),
    )
    # 新增：单次分析流程结束提示（防止用户等待中断后无反馈）
    click.echo("\n🚀 目录分析任务已完整结束")
    click.echo(f"🧾 可使用: mas results {run_id} 查看详情或在交互模式再次 /analyze 其他目录。")
    click.echo("—— 分析结束 ——")

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
            print("你> ", end="", flush=True)
            user = await asyncio.to_thread(lambda: input().strip())
        except (EOFError, KeyboardInterrupt):
            print("\n👋 检测到退出信号，结束会话。")
            break
        if not user:
            continue
        low = user.lower()
        if low in {"/exit", "/quit", "q", "quit", "exit"}:
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
                    await _async_wait_for_reports(agent_system, run_id, total_files)
                    # 明确结束通知
                    print(f"🎯 分析结束 run_id={run_id} ✅ 输入: results {run_id} 查看详情 或继续输入新的指令。")
                elif status == 'empty':
                    print("⚠️ 目录中未找到可分析的Python文件")
                else:
                    print(f"❌ 分析失败: {result.get('message','未知错误')}")
            except Exception as e:
                print(f"❌ 分析异常: {e}")
            continue
        
        # 处理可能的 JSON 导入请求 (硬编码路径识别)
        if user.endswith(".json") and os.path.exists(user):
            print(f"📦 检测到本地 JSON 文件路径: {user}")
            print("🧱 正在执行硬编码导入逻辑 (跳过 LLM)...")
            from utils.database_ingest import DatabaseIngestTool
            try:
                tool = DatabaseIngestTool()
                try:
                    await tool.process_file(user)
                    print("✅ 导入成功。")
                finally:
                    tool.close()
            except Exception as e:
                print(f"❌ 导入失败: {e}")
            continue

        # 普通对话消息
        resp = await agent_system.process_message_from_cli(user)
        if not resp.startswith("✅"):
            print(resp)

async def _login_entry(target_dir, use_cpu, output_dir=None):
    global _use_cpu_mode
    _use_cpu_mode = use_cpu

    click.echo("\n=====================================")
    click.echo("      MultiAgentSystem (MAS)")
    click.echo("      AI Code Review Assistant")
    click.echo(f"{'      CPU Mode' if use_cpu else '      GPU Mode'}")
    click.echo("=====================================")
    click.echo(f"Login successful at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if target_dir:
        if not os.path.isdir(target_dir):
            click.echo(f"❌ 目录不存在: {target_dir}")
            return
        target_dir = str(Path(target_dir).resolve())
        click.echo(f"📂 目标目录: {target_dir}")
        if output_dir:
            try:
                from utils.run_output import normalize_output_dir
                output_dir = normalize_output_dir(output_dir)
            except ValueError as e:
                click.echo(f"❌ {e}")
                return
            click.echo(f"📁 自定义输出父目录: reports/analysis/{output_dir}/<run_id>/")
        await _run_single_analysis_flow(target_dir, output_dir=output_dir)
        # 可选：分析后进入交互
        click.echo("📥 分析流程结束，进入交互会话。输入 /exit 退出。")
        agent_system = await _init_system()
        try:
            await _interactive_chat(agent_system)
        finally:
            # 清理资源，关闭连接
            await agent_system.shutdown_system()
    else:
        click.echo("未提供 --target-dir，仅初始化系统供后续交互。")
        agent_system = await _init_system()
        click.echo("系统初始化完成。可使用 '/analyze <dir>' 或命令行 'mas login -d <dir>' 分析。")
        # 进入持久交互循环
        try:
            await _interactive_chat(agent_system)
        finally:
            # 清理资源，关闭连接
            await agent_system.shutdown_system()

@mas.command()
@click.option('--target-dir', '-d', help='Directory containing code to review')
@click.option(
    '--output-dir',
    '-o',
    default=None,
    help='Parent folder under reports/analysis/; final path is reports/analysis/<output-dir>/<run_id>/',
)
@click.option('--cpu', is_flag=True, help='Init system in CPU mode (no GPU usage)')
def login(target_dir, output_dir, cpu):
    """系统加载及其初始化"""
    try:
        asyncio.run(_login_entry(target_dir, cpu, output_dir=output_dir))
    except KeyboardInterrupt:
        # Best-effort graceful shutdown on Ctrl+C
        try:
            from core.agents_integration import get_agent_integration_system
            agent_system = get_agent_integration_system()
            # Try synchronous disconnect of all Weaviate clients if present
            _disconnect_agent_vector_services(agent_system)
            # Attempt async shutdown (safe scheduling if loop running)
            try:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except Exception:
                    loop = None

                if loop is not None and loop.is_running():
                    try:
                        loop.call_soon_threadsafe(lambda: asyncio.create_task(agent_system.shutdown_system()))
                    except Exception:
                        import threading
                        threading.Thread(target=lambda: asyncio.run(agent_system.shutdown_system()), daemon=True).start()
                else:
                    try:
                        asyncio.run(agent_system.shutdown_system())
                    except Exception:
                        pass
            except Exception:
                pass
        finally:
            raise

async def _run_batch_flow(config_path: str, use_cpu: bool):
    """批量分析：从 JSON 配置读取多个 {target_dir, output_dir} 逐项分析，不进入交互。"""
    global _use_cpu_mode
    _use_cpu_mode = use_cpu

    config_path = str(Path(config_path).resolve())
    if not os.path.isfile(config_path):
        click.echo(f"❌ 配置文件不存在: {config_path}")
        return

    try:
        with open(config_path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        click.echo(f"❌ 读取配置文件失败: {e}")
        return

    # 支持 {"items": [...]} 或裸数组
    if isinstance(data, dict):
        items = data.get('items', [])
    elif isinstance(data, list):
        items = data
    else:
        items = []

    if not items:
        click.echo("⚠️ 配置文件中没有可分析的条目（期望 {\"items\": [{\"target_dir\": ..., \"output_dir\": ...}, ...]}）。")
        return

    click.echo(f"📋 批量分析：共 {len(items)} 项")
    agent_system = await _init_system()

    results = []
    try:
        for idx, item in enumerate(items, 1):
            if not isinstance(item, dict):
                click.echo(f"  [{idx}/{len(items)}] ⚠️ 跳过：条目不是对象")
                continue
            target = item.get('target_dir') or item.get('target-dir')
            output = item.get('output_dir') or item.get('output-dir')
            if not target:
                click.echo(f"  [{idx}/{len(items)}] ⚠️ 跳过：缺少 target_dir")
                continue
            target = str(Path(target).resolve())
            click.echo(f"\n  [{idx}/{len(items)}] 📂 {target}")
            if not os.path.isdir(target):
                click.echo(f"    ❌ 目录不存在，跳过")
                results.append({'target_dir': target, 'status': 'missing_dir'})
                continue
            if output:
                try:
                    from utils.run_output import normalize_output_dir
                    output = normalize_output_dir(output)
                except ValueError as e:
                    click.echo(f"    ❌ output_dir 无效: {e}，跳过")
                    results.append({'target_dir': target, 'status': 'bad_output'})
                    continue

            try:
                dispatch = await _dispatch_directory_analysis(agent_system, target, output_dir=output)
            except Exception as e:
                click.echo(f"    ❌ 派发异常: {e}")
                results.append({'target_dir': target, 'status': 'dispatch_error'})
                continue

            if dispatch.get('status') != 'dispatched':
                click.echo(f"    ❌ 派发失败: {dispatch}")
                results.append({'target_dir': target, 'status': 'not_dispatched'})
                continue

            run_id = dispatch['run_id']
            click.echo(f"    🆔 Run ID: {run_id}（已派发 {dispatch.get('total_files')} 个文件）")
            await _async_wait_for_reports(
                agent_system,
                run_id,
                dispatch.get('total_files'),
                timeout=dispatch.get('estimated_timeout_seconds'),
            )
            results.append({'target_dir': target, 'run_id': run_id, 'status': 'done'})
            click.echo(f"    ✅ 完成")
    finally:
        try:
            await agent_system.shutdown_system()
        except Exception:
            pass

    done = sum(1 for r in results if r.get('status') == 'done')
    failed = len(results) - done
    click.echo("\n" + "=" * 50)
    click.echo(f"📊 批量分析结束：成功 {done}，失败/跳过 {failed}，共 {len(results)}")
    for r in results:
        if r.get('status') != 'done':
            click.echo(f"  ❌ {r.get('target_dir')} -> {r.get('status')}")
    click.echo("=" * 50)

    # 自动汇总：批结束后生成逐项 CSV；400 实验批再自动跑评测并出汇总表。
    try:
        from utils.experiments.auto_summary import run_auto_summary
        run_auto_summary(config_path, items, results)
    except Exception as e:
        click.echo(f"⚠️ 自动汇总失败(不影响批量分析结果): {e}")


@mas.command()
@click.option('--config', '-c', required=True, help='Batch JSON: {"items":[{"target_dir":..., "output_dir":...}, ...]}')
@click.option('--cpu', is_flag=True, help='CPU mode')
def batch(config, cpu):
    """批量分析：读取 JSON 配置，逐项分析目录（不进入交互）"""
    try:
        asyncio.run(_run_batch_flow(config, cpu))
    except KeyboardInterrupt:
        raise


if __name__ == '__main__':
    mas()

main = mas