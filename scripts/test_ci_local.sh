#!/bin/bash
# Local CI validation script - mirrors GitHub Actions CI workflow

set -e

echo "🧪 Running local CI validation..."
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2 passed${NC}"
    else
        echo -e "${RED}❌ $2 failed${NC}"
        exit 1
    fi
}

echo -e "${YELLOW}1. Checking ruff linting...${NC}"
ruff check . --exclude '*.ipynb'
print_status $? "Ruff linting"

echo -e "${YELLOW}2. Checking ruff formatting...${NC}"
ruff format --check . --exclude '*.ipynb'
print_status $? "Ruff formatting"

echo -e "${YELLOW}3. Running mypy type checking...${NC}"
mypy optimal_cutoffs/
print_status $? "MyPy type checking"

echo -e "${YELLOW}4. Running pytest...${NC}"
if command -v pytest-cov &> /dev/null; then
    pytest --cov=optimal_cutoffs --cov-report=term-missing -q
    print_status $? "Tests with coverage"
else
    pytest -q
    print_status $? "Tests"
fi

echo -e "${GREEN}🎉 All CI checks passed locally!${NC}"
echo "Your code is ready for GitHub CI/CD"