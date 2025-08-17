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

    click.echo("AI assistant is ready. Type your questions or commands (type 'exit' to quit).")
    click.echo("Note: agent system integration is available but currently decoupled.\n")

    start_conversation(validated_target_dir)

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
        
        # 只在关键错误时进行基本检查
        core_path = os.path.join(project_root, 'core')
        if not os.path.exists(core_path):
            logger.error(f"❌ core目录不存在: {core_path}")
            
        click.echo("❌ 多智能体分析系统不可用")
        click.echo(f"导入错误: {e}")
        
    except Exception as e:
        logger.error(f"❌ 智能体系统初始化错误: {e}")
        
        click.echo("❌ 多智能体分析系统初始化失败")
        click.echo(f"错误: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        
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

            # 转发给智能体系统进行处理
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
                    click.echo(f"❌ 智能体系统错误: {e}")
            else:
                # 如果智能体系统不可用，显示AI响应
                ai_response = generate_ai_response(user_input, target_dir)
                click.echo(f"AI: {ai_response}")

            click.echo()
        except KeyboardInterrupt:
            click.echo("\nThank you for using MAS. Goodbye!")
            break
        except Exception as e:
            logger.error(f"主循环错误: {e}")
            click.echo(f"An error occurred: {str(e)}", err=True)

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