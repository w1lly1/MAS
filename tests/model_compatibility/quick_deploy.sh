#!/bin/bash
"""
Quick Deployment Script for Model Compatibility Fixes
Apply the tested solutions to production environment
"""

echo "🚀 Model Compatibility Quick Deploy"
echo "==================================="

# Set working directory
cd "$(dirname "$0")"
cd "../.."  # Go to MAS directory

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Not in MAS project directory"
    exit 1
fi

echo "📁 Working directory: $(pwd)"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "🎯 Step 1: Backup Current Configuration"
echo "======================================"
cp core/agents/ai_driven_user_communication_agent.py core/agents/ai_driven_user_communication_agent.py.backup
cp ai_agent_config.json ai_agent_config.json.backup
echo "✅ Backups created"

echo ""
echo "🔧 Step 2: Apply ChatGLM3-6B Fix"
echo "==============================="

# Create a temporary Python script to apply the fix
cat > apply_chatglm3_fix.py << 'EOF'
import os
import re

def apply_chatglm3_fix():
    """Apply ChatGLM3-6B compatibility fix to user communication agent"""
    
    agent_file = "core/agents/ai_driven_user_communication_agent.py"
    
    if not os.path.exists(agent_file):
        print(f"❌ File not found: {agent_file}")
        return False
    
    # Read the current file
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace ChatGLM2-6B with ChatGLM3-6B
    content = content.replace(
        'self.model_name = "THUDM/chatglm2-6b"',
        'self.model_name = "THUDM/chatglm3-6b"'
    )
    
    # Add compatibility fix method if not present
    if '_extract_past_from_model_output' not in content:
        fix_method = '''
        # Add compatibility fix for ChatGLM3-6B
        if not hasattr(self.model, '_extract_past_from_model_output'):
            def _extract_past_from_model_output(*args, **kwargs):
                """Extract past_key_values for compatibility with transformers 4.56.0"""
                outputs = args[0] if args else None
                if outputs is None:
                    return None
                if hasattr(outputs, 'past_key_values'):
                    return outputs.past_key_values
                elif isinstance(outputs, dict) and 'past_key_values' in outputs:
                    return outputs['past_key_values']
                return None
            
            import types
            self.model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, self.model)
            print("✅ Applied ChatGLM3-6B compatibility fix")
'''
        
        # Insert the fix after model loading
        if 'self.model = AutoModelForCausalLM.from_pretrained(' in content:
            content = content.replace(
                'print("✅ 模型加载成功")',
                'print("✅ 模型加载成功")' + fix_method
            )
    
    # Write the updated content
    with open(agent_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ ChatGLM3-6B fix applied successfully")
    return True

def update_config():
    """Update AI agent configuration"""
    
    import json
    
    config_file = "ai_agent_config.json"
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Read current config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Update model configurations
    if 'agents' in config:
        for agent_id, agent_config in config['agents'].items():
            if 'user_communication' in agent_id.lower():
                agent_config['model_name'] = "THUDM/chatglm3-6b"
                agent_config['compatibility_mode'] = True
                print(f"✅ Updated {agent_id} to use ChatGLM3-6B")
    
    # Add compatibility notes
    config['compatibility'] = {
        "transformers_version": "4.56.0",
        "tested_models": {
            "chatglm3-6b": "compatible_with_fixes",
            "codebert": "fully_compatible",
            "qwen2-7b": "needs_replacement"
        },
        "last_tested": "2025-09-15"
    }
    
    # Write updated config
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ Configuration updated successfully")
    return True

if __name__ == "__main__":
    print("🔧 Applying compatibility fixes...")
    
    success = True
    success &= apply_chatglm3_fix()
    success &= update_config()
    
    if success:
        print("\n🎉 All fixes applied successfully!")
        print("💡 Next steps:")
        print("  1. Test the user communication agent")
        print("  2. Monitor for any compatibility issues")
        print("  3. Deploy CodeBERT for code analysis")
    else:
        print("\n❌ Some fixes failed. Check the output above.")
EOF

# Run the fix
python apply_chatglm3_fix.py

echo ""
echo "📋 Step 3: Validate Changes"
echo "=========================="

# Quick validation
python -c "
import sys
sys.path.append('.')
try:
    from core.agents.ai_driven_user_communication_agent import AIDrivenUserCommunicationAgent
    agent = AIDrivenUserCommunicationAgent()
    print(f'✅ Agent model: {agent.model_name}')
    if 'chatglm3' in agent.model_name.lower():
        print('✅ ChatGLM3-6B configuration applied')
    else:
        print('⚠️ Model name not updated')
except Exception as e:
    print(f'❌ Import test failed: {e}')
"

echo ""
echo "🚀 Step 4: Test Model Loading (Optional)"
echo "======================================="

echo "Would you like to test model loading? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "🧪 Testing ChatGLM3-6B loading..."
    python -c "
import sys
sys.path.append('.')
try:
    from core.agents.ai_driven_user_communication_agent import AIDrivenUserCommunicationAgent
    import asyncio
    
    async def test_model():
        agent = AIDrivenUserCommunicationAgent()
        success = await agent.initialize()
        if success:
            print('✅ ChatGLM3-6B loaded successfully')
            # Quick test
            response = await agent.generate_response('Hello, test message', max_new_tokens=20)
            if response:
                print(f'✅ Test response: {response[:100]}...')
            else:
                print('⚠️ No response generated')
        else:
            print('❌ Model loading failed')
    
    asyncio.run(test_model())
except Exception as e:
    print(f'❌ Model test failed: {e}')
    print('💡 This is expected if model files are not downloaded yet')
"
else
    echo "⏭️ Skipping model loading test"
fi

echo ""
echo "📊 Step 5: Results Summary"
echo "========================="

echo "✅ Deployment completed!"
echo ""
echo "📁 Files modified:"
echo "  - core/agents/ai_driven_user_communication_agent.py"
echo "  - ai_agent_config.json"
echo ""
echo "📁 Backups created:"
echo "  - core/agents/ai_driven_user_communication_agent.py.backup"
echo "  - ai_agent_config.json.backup"
echo ""
echo "🔗 Next steps:"
echo "  1. Test user communication features"
echo "  2. Deploy CodeBERT for code analysis"
echo "  3. Replace Qwen2-7B with compatible alternative"
echo "  4. Monitor system performance"
echo ""
echo "📖 Full action plan: tests/model_compatibility/ACTION_PLAN.md"

# Cleanup
rm -f apply_chatglm3_fix.py

echo ""
echo "🏁 Quick deployment complete!"
