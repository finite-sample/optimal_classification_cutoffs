#!/bin/bash
# Comprehensive CI Simulation Script
# This script runs the exact same steps that CI runs to catch issues locally

set -e  # Exit on any error

echo "🔄 Running Comprehensive CI Simulation"
echo "====================================="
echo "This matches the exact CI pipeline to catch issues locally"
echo ""

# Step 1: Install package in development mode
echo "📦 Step 1: Installing package in development mode..."
python3 -m pip install -e . --break-system-packages || {
    echo "❌ Package installation failed"
    echo "You may need to run this in a virtual environment"
    exit 1
}
echo "✅ Package installed successfully"
echo ""

# Step 2: Lint check (exactly like CI) 
echo "🧹 Step 2: Running linting (matches CI)..."
ruff check . --exclude '*.ipynb' || {
    echo "❌ Linting failed - fix these issues before committing"
    exit 1  
}
echo "✅ Linting passed"
echo ""

# Step 3: Full test collection (catches import errors)
echo "🔍 Step 3: Testing full test collection (matches CI)..."
python3 -m pytest --collect-only -q > /dev/null || {
    echo "❌ Test collection failed - there are import errors"
    echo "Running again with verbose output:"
    python3 -m pytest --collect-only -q
    exit 1
}
echo "✅ Test collection successful - no import errors"  
echo ""

# Step 4: Type checking (if mypy is available)
echo "🔧 Step 4: Type checking (optional)..."
if command -v mypy &> /dev/null; then
    mypy optimal_cutoffs/ --ignore-missing-imports || {
        echo "⚠️  Type checking found issues (non-blocking)"
    }
    echo "✅ Type checking completed"
else
    echo "⚠️  mypy not installed - skipping type check"
fi
echo ""

# Step 5: Run full test suite with coverage (like CI) 
echo "🧪 Step 5: Running full test suite..."
echo "This may take a while - running all tests like CI does..."
python3 -m pytest tests/ -x --tb=short || {
    echo "❌ Tests failed - fix these before committing"
    exit 1
}
echo "✅ All tests passed"
echo ""

# Step 6: Check for any uncommitted changes
echo "📋 Step 6: Checking git status..."
if [[ -n $(git status --porcelain) ]]; then
    echo "⚠️  There are uncommitted changes:"
    git status --porcelain
    echo ""
    echo "Consider committing these changes"
else
    echo "✅ No uncommitted changes"
fi
echo ""

echo "🎉 All CI simulation steps passed!"
echo "✅ Your code should pass CI - safe to commit and push"
echo ""
echo "To run this again: ./scripts/ci-simulation.sh"