import click
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# 配置日志，只在错误时显示详细信息
logging.basicConfig(
    level=logging.WARNING,  # 只显示WARNING以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def mas():
    """MultiAgentSystem (MAS) - AI代码审查助手"""
    pass

@mas.command()
@click.option('--target-dir', '-d', help='Directory containing code to review')
def login(target_dir):
    """Login to MAS system and start AI conversation"""
    
    # 检查是否为管道输入（在输出任何信息之前）
    is_interactive = sys.stdin.isatty()
    piped_input = None
    
    if not is_interactive:
        try:
            # 读取管道输入的所有内容
            piped_input = sys.stdin.read().strip()
        except Exception as e:
            click.echo(f"❌ 读取管道输入时出错: {e}")
            return
    
    click.echo("\n=====================================")
    click.echo("      MultiAgentSystem (MAS)")
    click.echo("      AI Code Review Assistant")
    click.echo("=====================================")
    click.echo(f"Login successful at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    validated_target_dir = None
    if target_dir:
        if not os.path.isdir(target_dir):
            click.echo(f"Error: Directory '{target_dir}' does not exist.", err=True)
            return
        validated_target_dir = str(Path(target_dir).resolve())
        click.echo(f"Monitoring code directory: {validated_target_dir}\n")
    else:
        click.echo("No target directory specified. Use --target-dir to set code review directory.\n")

    # 如果有管道输入，直接处理并退出
    if piped_input:
        click.echo(f"📥 接收到输入: {piped_input}")
        start_conversation_with_input(validated_target_dir, piped_input)
    else:
        click.echo("AI assistant is ready. Type your questions or commands (type 'exit' to quit).")
        click.echo("Note: agent system integration is available but currently decoupled.\n")
        start_conversation(validated_target_dir)

@mas.command()
def config():
    """Configure AI agent system settings"""
    click.echo("\n🤖 AI智能体系统配置")
    
    # 确保项目根目录在Python路径中
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
                click.echo("\n🔍 测试AI模型连接...")
                click.echo("注意: 需要在运行状态下测试，请使用 'mas status' 命令")
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
def status():
    """Check AI agent system status"""
    click.echo("\n🔍 检查AI智能体系统状态...")
    
    # 确保项目根目录在Python路径中
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from core.agents_integration import get_agent_integration_system
        from core.ai_agent_config import get_ai_agent_config
        
        agent_system = get_agent_integration_system()
        config_manager = get_ai_agent_config()
        
        # 配置状态
        click.echo("\n📋 配置状态:")
        summary = config_manager.get_config_summary()
        click.echo(f"运行模式: {summary['agent_mode']}")
        click.echo(f"配置有效: {'✅' if summary['config_valid'] else '❌'}")
        
        # 尝试初始化并检查状态
        if not agent_system._system_ready:
            click.echo("\n🔧 初始化智能体系统...")
            asyncio.run(agent_system.initialize_system())
        
        # 智能体状态
        status = asyncio.run(agent_system.get_agent_status())
        click.echo(f"\n🤖 系统状态: {'✅ 就绪' if status['system_ready'] else '❌ 未就绪'}")
        
        click.echo("\n📋 智能体列表:")
        active_agents = agent_system.get_active_agents()
        for name, class_name in active_agents.items():
            ai_indicator = "🤖" if name.startswith('ai_') else "🔧"
            click.echo(f"  {ai_indicator} {name}: {class_name}")
        
        # 测试AI智能体
        if any(name.startswith('ai_') for name in active_agents.keys()):
            click.echo("\n🧪 测试AI智能体...")
            ai_test_results = asyncio.run(agent_system.test_ai_agents())
            
            for agent_name, result in ai_test_results.items():
                status_icon = "✅" if result['status'] == 'available' else "❌"
                ai_status = result.get('ai_ready', 'unknown')
                ai_icon = "🤖" if ai_status else "⚠️"
                click.echo(f"  {status_icon} {ai_icon} {agent_name}: {result['status']}")
                if 'error' in result:
                    click.echo(f"    错误: {result['error']}")
        
        click.echo(f"\n📊 总计: {len(active_agents)} 个AI智能体已加载")
        
    except Exception as e:
        click.echo(f"❌ 状态检查失败: {e}")
        import traceback
        logger.error(f"状态检查错误: {traceback.format_exc()}")

def start_conversation_with_input(target_dir=None, user_input=None):
    """处理管道输入的对话"""
    # 初始化智能体系统
    agent_system = None
    
    # 确保项目根目录在Python路径中
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from core.agents_integration import get_agent_integration_system
        
        agent_system = get_agent_integration_system()
        
        # 初始化系统
        asyncio.run(agent_system.initialize_system())
        
        click.echo("🤖 多智能体分析系统已加载并准备集成")
        
        # 处理用户输入
        if user_input and agent_system:
            try:
                result = asyncio.run(
                    agent_system.process_message_from_cli(user_input, target_dir)
                )
                if not result.startswith("✅"):
                    click.echo(f"🤖 {result}")
            except Exception as e:
                logger.error(f"❌ 智能体系统处理输入错误: {e}")
                click.echo(f"❌ 系统错误: {e}")
        
        # 处理完成，关闭系统
        click.echo("📋 分析任务已完成，程序退出")
        if agent_system:
            try:
                asyncio.run(agent_system.shutdown_system())
            except Exception as e:
                logger.error(f"关闭智能体系统时出错: {e}")
                
    except Exception as e:
        logger.error(f"❌ 智能体系统初始化错误: {e}")
        click.echo("❌ 多智能体分析系统初始化失败")
        click.echo(f"错误: {e}")

def start_conversation(target_dir=None):
    """Start interactive conversation with AI model"""
    # 初始化智能体系统
    agent_system = None
    
    # 确保项目根目录在Python路径中
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from core.agents_integration import get_agent_integration_system
        
        agent_system = get_agent_integration_system()
        
        # 初始化系统
        asyncio.run(agent_system.initialize_system())
        
        click.echo("🤖 多智能体分析系统已加载并准备集成")
        
    except ImportError as e:
        logger.error(f"❌ 导入错误: {e}")
        click.echo("❌ 多智能体分析系统不可用")
        click.echo(f"导入错误: {e}")
        
    except Exception as e:
        logger.error(f"❌ 智能体系统初始化错误: {e}")
        click.echo("❌ 多智能体分析系统初始化失败")
        click.echo(f"错误: {e}")

    # 如果有目标目录，转发给智能体系统
    if target_dir and agent_system:
        try:
            logger.debug(f"向智能体系统发送消息: 请分析目录: {target_dir}")
            result = asyncio.run(
                agent_system.process_message_from_cli(
                    f"请分析目录: {target_dir}", target_dir
                )
            )
            logger.debug(f"智能体系统响应: {result}")
            click.echo(f"🔄 智能体系统: {result}")
        except Exception as e:
            logger.error(f"❌ 智能体系统处理消息错误: {e}")
            click.echo(f"❌ 智能体系统错误: {e}")

    # 交互模式主循环
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                # 清理智能体系统
                if agent_system:
                    try:
                        asyncio.run(agent_system.shutdown_system())
                    except Exception as e:
                        logger.error(f"关闭智能体系统时出错: {e}")
                        
                click.echo("Thank you for using MAS. Goodbye!")
                break

            # 智能体系统处理用户输入
            if agent_system and user_input.strip():
                try:
                    result = asyncio.run(
                        agent_system.process_message_from_cli(user_input, target_dir)
                    )
                    # 智能体系统已经直接输出结果，这里只显示状态
                    if not result.startswith("✅"):
                        click.echo(f"🤖 {result}")
                except Exception as e:
                    logger.error(f"❌ 智能体系统处理用户输入错误: {e}")
                    click.echo(f"❌ 系统错误: {e}")
                    click.echo("💡 请尝试重新输入或使用 'help' 查看使用指南")
            else:
                # 如果智能体系统不可用，提供基本指导
                click.echo("❌ 智能体系统不可用")
                click.echo("💡 请输入 'help' 查看使用指南，或重启系统")

            click.echo()
        except KeyboardInterrupt:
            click.echo("\nThank you for using MAS. Goodbye!")
            break
        except EOFError:
            # 处理EOF错误（Ctrl+D或管道结束）
            click.echo("\n📋 输入结束，程序退出")
            break
        except Exception as e:
            logger.error(f"主循环错误: {e}")
            click.echo(f"An error occurred: {str(e)}", err=True)
            # 如果是EOF相关错误，退出循环
            if "EOF" in str(e):
                click.echo("📋 输入流结束，程序退出")
                break

def generate_ai_response(user_input, target_dir=None):
    """Simulate AI response for CLI interface"""
    default_agent_message = f"命令行转发: {user_input}"
    if target_dir:
        default_agent_message += f" (目标目录: {target_dir})"
    
    responses = {
        "hello": "Hello! I'm your MAS AI assistant. How can I help you with code review today?",
        "help": "I can help with code quality analysis, security checks, and performance reviews. You can specify a directory with --target-dir.",
        "what can you do": "I can analyze code quality, detect security vulnerabilities, and provide improvement suggestions using multiple AI agents.",
        "analyze code": f"Please specify a target directory using the --target-dir option. Current target: {target_dir or 'None'}",
        "agent status": "多智能体分析系统已加载，包含静态扫描、代码质量、安全分析和性能分析智能体。",
    }
    
    agent_note = f" [将转发给多智能体系统: '{default_agent_message}']"
    
    base_response = responses.get(user_input.lower(), 
                      f"I'm processing your request: '{user_input}'. The multi-agent system will provide detailed analysis.")
    
    return base_response + agent_note

if __name__ == '__main__':
    mas()

main = mas