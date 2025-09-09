#!/bin/bash
"""
Model Compatibility Test Runner
Automated script to run compatibility tests
"""

echo "🚀 Model Compatibility Test Runner"
echo "=================================="

# Set working directory
cd "$(dirname "$0")"
cd "../.."  # Go to MAS directory

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Not in MAS project directory"
    exit 1
fi

echo "📁 Working directory: $(pwd)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Create results directory
mkdir -p tests/model_compatibility/results

echo ""
echo "🔍 Step 1: Environment Check"
echo "============================"
python tests/model_compatibility/environment_check.py

echo ""
echo "⚡ Step 2: Quick Compatibility Check"
echo "=================================="
python tests/model_compatibility/quick_check.py

echo ""
echo "🧪 Step 3: Full Compatibility Test Suite"
echo "======================================="
python tests/model_compatibility/compatibility_test_suite.py

echo ""
echo "📊 Step 4: Results Summary"
echo "========================="

# Check if results exist and show summary
if [ -f "tests/model_compatibility/results/compatibility_report.json" ]; then
    echo "✅ Compatibility tests completed successfully"
    echo ""
    echo "📁 Results available in:"
    echo "  - tests/model_compatibility/results/environment_report.json"
    echo "  - tests/model_compatibility/results/compatibility_report.json"
    echo "  - tests/model_compatibility/results/detailed_results.json"
    echo ""
    echo "🔗 Next steps:"
    echo "  1. Review the compatibility report"
    echo "  2. Address any compatibility issues"
    echo "  3. Run individual model tests for specific debugging"
    echo "  4. Deploy compatible models to production"
else
    echo "❌ Compatibility tests may have failed"
    echo "Check the output above for errors"
fi

echo ""
echo "🏁 Model compatibility testing complete!"
