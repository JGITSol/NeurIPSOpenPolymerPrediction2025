# Contributing to Polymer Prediction

We welcome contributions to the Polymer Prediction project! This document provides guidelines for contributing.

## Development Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd polymer-prediction
    ```

2.  Set up the development environment:

    ```bash
    make setup-env
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install pre-commit hooks:

    ```bash
    pre-commit install
    ```

## Code Style

We use several tools to maintain code quality:

-   **Black** for code formatting
-   **isort** for import sorting
-   **flake8** for linting
-   **mypy** for type checking
-   **bandit** for security scanning

Run all checks with:

```bash
make lint
make type-check
make security
```

Or format code automatically:

```bash
make format
```

## Testing

We use pytest for testing. Run tests with:

```bash
make test
```

For faster testing without coverage:

```bash
make test-fast
```

## Pull Request Process

1.  Fork the repository
2.  Create a feature branch: `git checkout -b feature/your-feature-name`
3.  Make your changes
4.  Add tests for new functionality
5.  Ensure all tests pass: `make test`
6.  Ensure code quality checks pass: `make lint type-check security`
7.  Commit your changes with a descriptive message
8.  Push to your fork: `git push origin feature/your-feature-name`
9.  Create a pull request

## Commit Messages

Use clear, descriptive commit messages:

-   Use the present tense ("Add feature" not "Added feature")
-   Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
-   Limit the first line to 72 characters or less
-   Reference issues and pull requests liberally after the first line

## Code Organization

-   `src/polymer_prediction/` - Main source code
-   `tests/` - Test files
-   `configs/` - Configuration files
-   `docs/` - Documentation
-   `scripts/` - Utility scripts

## Documentation

-   Update docstrings for new functions and classes
-   Follow Google-style docstrings
-   Update README.md if adding new features
-   Add examples for new functionality

## Issue Reporting

When reporting issues, please include:

-   Python version
-   Operating system
-   Steps to reproduce
-   Expected behavior
-   Actual behavior
-   Error messages (if any)

## Feature Requests

For feature requests, please:

-   Check if the feature already exists
-   Describe the use case
-   Explain why it would be beneficial
-   Consider implementation complexity

Thank you for contributing!