# Contributing to NVIDIA Blog MCP Server

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Include clear descriptions, steps to reproduce, and expected vs actual behavior
- Check existing issues first to avoid duplicates

### Pull Requests

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the code style guidelines
3. **Test your changes** to ensure they work correctly
4. **Update documentation** if you've changed functionality
5. **Submit a pull request** with a clear description of changes

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SynapGarden/NVIDIA_blog_mcp.git
   cd NVIDIA_blog_mcp
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have GCP authentication configured:
   ```bash
   gcloud auth application-default login
   ```

## Code Style

- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

- Test your changes locally before submitting
- Ensure the MCP server starts without errors
- Test query functionality if you've modified query modules

## Documentation

- Update README.md if you've added features or changed setup
- Update docstrings in code for new functions/classes
- Keep inline code comments clear and helpful

## Commit Messages

- Use clear, descriptive commit messages
- Reference issue numbers when applicable (e.g., "Fix #123: ...")
- Keep commits focused on a single change when possible

## Questions?

Feel free to open an issue with questions or reach out to maintainers via GitHub Issues.

Thank you for contributing!
