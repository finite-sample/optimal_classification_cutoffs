# Test Suite Organization

This directory contains the comprehensive test suite for the optimal_cutoffs library. The tests are organized into logical categories to facilitate development, debugging, and CI/CD processes.

## Directory Structure

```
tests/
├── fixtures/           # Shared test utilities and data generators
├── unit/              # Unit tests for individual components  
├── integration/       # Integration tests for end-to-end workflows
├── edge_cases/        # Edge cases and boundary condition tests
├── validation/        # Input validation and error handling tests
├── performance/       # Performance and scalability tests
├── slow/             # Long-running comprehensive tests
├── conftest.py       # Pytest configuration and shared fixtures
├── pytest.ini       # Pytest settings and markers
└── README.md         # This file
```

## Running Tests

### Basic Usage
```bash
pytest tests/                    # Run all tests
pytest tests/unit/               # Run unit tests
pytest -m "not slow"             # Exclude slow tests
pytest tests/slow/ --runslow     # Run slow tests (explicit flag required)
```

## Test Categories

- **Unit Tests**: Fast tests for individual components
- **Integration Tests**: End-to-end workflow testing
- **Edge Cases**: Boundary conditions and extreme scenarios
- **Validation**: Input validation and error handling
- **Performance**: Algorithmic complexity and benchmarking
- **Slow Tests**: Comprehensive scenarios requiring extended time

## Development

Use shared fixtures from `conftest.py` and data generators from `fixtures/` when writing new tests. Follow the established patterns for test organization and marking.