# Contributing to Tree-PIC → QPC

Thank you for your interest in contributing to the Tree-PIC → QPC library! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12.0 or higher
- Git

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pic-qpc.git
   cd pic-qpc
   ```
3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking

Run these tools before committing:

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy pic/
```

### Testing

Write tests for new functionality:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pic --cov-report=html

# Run specific test file
pytest tests/test_your_module.py

# Run with verbose output
pytest -v
```

### Documentation

Update documentation for new features:

- **README.md**: User-facing documentation
- **docs/design.md**: Architecture and implementation details
- **docs/math.md**: Mathematical foundations
- **docs/experiments.md**: Experimental results
- **Docstrings**: Inline documentation for functions and classes

## Pull Request Process

1. **Ensure tests pass**:
   ```bash
   pytest tests/
   ```

2. **Check code quality**:
   ```bash
   black --check .
   ruff check .
   mypy pic/
   ```

3. **Update documentation** if needed

4. **Create a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Test results
   - Any breaking changes

5. **Wait for review** and address feedback

## Code Review Guidelines

### For Contributors

- Keep PRs focused and reasonably sized
- Include tests for new functionality
- Update documentation as needed
- Respond promptly to review comments

### For Reviewers

- Be constructive and respectful
- Focus on code quality and correctness
- Check for test coverage
- Verify documentation updates

## Issue Reporting

When reporting issues, please include:

- **Description**: Clear description of the problem
- **Reproduction**: Steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, PyTorch version, OS
- **Code example**: Minimal code to reproduce the issue

## Feature Requests

For feature requests:

- **Description**: Clear description of the feature
- **Use case**: Why this feature is needed
- **Proposed implementation**: How you think it should work
- **Alternatives**: Any alternatives you've considered

## Testing Guidelines

### Unit Tests

- Test individual functions and classes
- Use descriptive test names
- Include edge cases and error conditions
- Mock external dependencies

### Integration Tests

- Test complete workflows
- Use realistic data
- Test error handling
- Verify end-to-end functionality

### Property-Based Tests

- Use Hypothesis for property-based testing
- Test invariants and properties
- Generate random test cases
- Test edge cases automatically

## Documentation Guidelines

### Code Documentation

- Use NumPy-style docstrings
- Include type hints
- Document parameters and return values
- Provide usage examples

### User Documentation

- Write for the target audience
- Include code examples
- Use clear, concise language
- Keep documentation up to date

## Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Version number is updated
- [ ] CHANGELOG is updated
- [ ] Release notes are written

## Community Guidelines

### Communication

- Be respectful and inclusive
- Use clear, constructive language
- Assume good intentions
- Be patient with newcomers

### Collaboration

- Help others learn and grow
- Share knowledge and expertise
- Give credit where due
- Support the community

## Getting Help

If you need help:

1. **Check the documentation** first
2. **Search existing issues** for similar problems
3. **Ask questions** in GitHub Discussions
4. **Open an issue** for bugs or feature requests

## Recognition

Contributors will be recognized in:

- **README.md**: List of contributors
- **Release notes**: Credit for contributions
- **Documentation**: Attribution for significant contributions

Thank you for contributing to Tree-PIC → QPC!
