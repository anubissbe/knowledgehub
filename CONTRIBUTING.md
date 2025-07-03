# Contributing to KnowledgeHub

We love your input\! We want to make contributing to KnowledgeHub as easy and transparent as possible, whether it is:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you have added code that should be tested, add tests.
3. If you have changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request\!

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/knowledgehub.git
cd knowledgehub

# Start development environment
docker compose up -d

# Install frontend dependencies
cd src/web-ui
npm install
npm run dev
```

### Code Style

- **Python**: Follow PEP 8, use `black` for formatting
- **TypeScript**: Use ESLint rules, follow React best practices
- **Documentation**: Update README.md for user-facing changes

### Testing

```bash
# Run Python tests
python test_all_functionality.py
python test_code_quality.py

# Run system integration tests
./test_complete_system.sh

# Frontend testing
cd src/web-ui && npm test
```

## Any contributions you make will be under the MIT Software License

When you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project.

## Report bugs using GitHub Issues

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/anubissbe/knowledgehub/issues/new).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific\!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn not work)

## Feature Requests

We welcome feature requests\! Please:

1. Check if the feature already exists or is planned
2. Open an issue describing the feature
3. Explain why this feature would be useful
4. Be willing to help implement it

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
