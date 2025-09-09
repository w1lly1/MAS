#!/bin/bash
# Deploy Alternative Models (Qwen1.5 replacement for Qwen2)

echo "🚀 Deploying Alternative Model Configuration"
echo "============================================="

# Activate virtual environment
source venv/bin/activate

# Install any missing dependencies
pip install tiktoken protobuf

# Test new configuration
echo "🧪 Testing new model configuration..."
python tests/model_compatibility/test_qwen15_replacement.py

# Run compatibility checks
echo "📊 Running compatibility checks..."
python tests/model_compatibility/compatibility_test_suite.py

# Deploy to production if tests pass
echo "🎯 Deploying to production environment..."
python quick_deploy.sh

echo "✅ Deployment complete!"
