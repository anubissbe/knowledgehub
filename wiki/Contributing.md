# Contributing to KnowledgeHub

Thank you for your interest in contributing to KnowledgeHub! This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat everyone with respect and kindness
- **Be collaborative**: Work together to solve problems
- **Be inclusive**: Welcome newcomers and help them get started
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone was new once

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**To report a bug:**

1. Use the [Bug Report Template](https://github.com/anubissbe/knowledgehub/issues/new?template=bug_report.md)
2. Include:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - System information
   - Relevant logs

### Suggesting Enhancements

**To suggest an enhancement:**

1. Use the [Feature Request Template](https://github.com/anubissbe/knowledgehub/issues/new?template=feature_request.md)
2. Include:
   - Clear use case
   - Proposed solution
   - Alternative solutions considered
   - Additional context

### Code Contributions

Areas where we especially welcome contributions:

- **New Features**: Implement features from our roadmap
- **Bug Fixes**: Fix reported issues
- **Performance**: Optimization and scaling improvements
- **Documentation**: Improve guides and API docs
- **Tests**: Increase test coverage
- **UI/UX**: Enhance the user interface

### Documentation Contributions

- Fix typos and clarify confusing sections
- Add examples and tutorials
- Translate documentation
- Create video tutorials
- Write blog posts about your experience

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker 24.0+
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/knowledgehub.git
cd knowledgehub
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/anubissbe/knowledgehub.git
```

### Environment Setup

1. **Python Environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Node.js Environment:**
```bash
cd src/web-ui
npm install
```

3. **Docker Services:**
```bash
docker compose up -d postgres redis weaviate-lite minio
```

4. **Environment Variables:**
```bash
cp .env.example .env.dev
# Edit .env.dev with your settings
```

### Running Tests

```bash
# Python tests
python -m pytest tests/

# JavaScript tests
cd src/web-ui
npm test

# Integration tests
./test_complete_system.sh
```

## Development Workflow

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# Or: git checkout -b fix/issue-number-description
```

### 2. Make Changes

- Write clean, readable code
- Follow coding standards
- Add tests for new features
- Update documentation

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new search filter feature

- Add filter by date range
- Add filter by content type
- Update API documentation
- Add unit tests

Closes #123"
```

**Commit Message Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/tool changes

### 4. Push Changes

```bash
git push origin feature/your-feature-name
```

## Coding Standards

### Python Code Style

We use Black for code formatting and follow PEP 8:

```bash
# Format code
black src/

# Check code style
flake8 src/

# Type checking
mypy src/
```

**Key Guidelines:**
- Use type hints
- Write docstrings for all functions/classes
- Keep functions small and focused
- Use meaningful variable names

**Example:**
```python
from typing import List, Dict, Optional
from pydantic import BaseModel

class SearchRequest(BaseModel):
    """Model for search API requests."""
    
    query: str
    search_type: str = "hybrid"
    limit: int = 20
    filters: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "query": "docker deployment",
                "search_type": "hybrid",
                "limit": 10
            }
        }

async def search_documents(
    request: SearchRequest,
    source_ids: Optional[List[str]] = None
) -> SearchResponse:
    """
    Search for documents across knowledge sources.
    
    Args:
        request: Search request parameters
        source_ids: Optional list of source IDs to search
        
    Returns:
        SearchResponse with matching documents
        
    Raises:
        ValueError: If search_type is invalid
    """
    # Implementation
    pass
```

### JavaScript/TypeScript Code Style

We use ESLint and Prettier:

```bash
# Format code
npm run format

# Lint code
npm run lint

# Type check
npm run type-check
```

**Key Guidelines:**
- Use TypeScript for type safety
- Prefer functional components
- Use React hooks appropriately
- Keep components small and reusable

**Example:**
```typescript
import React, { useState, useCallback } from 'react';
import { SearchResult } from '../types';

interface SearchResultsProps {
  results: SearchResult[];
  loading: boolean;
  onResultClick: (result: SearchResult) => void;
}

export const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  loading,
  onResultClick,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  
  const handleClick = useCallback((result: SearchResult) => {
    setSelectedId(result.id);
    onResultClick(result);
  }, [onResultClick]);
  
  if (loading) {
    return <div className="loading">Searching...</div>;
  }
  
  return (
    <div className="search-results">
      {results.map((result) => (
        <SearchResultItem
          key={result.id}
          result={result}
          selected={result.id === selectedId}
          onClick={() => handleClick(result)}
        />
      ))}
    </div>
  );
};
```

### Docker Best Practices

- Use specific base image versions
- Minimize layers
- Use multi-stage builds
- Run as non-root user
- Include health checks

**Example:**
```dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── fixtures/      # Test data
```

### Writing Tests

**Python Tests (pytest):**
```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_api_client():
    """Fixture for mocked API client."""
    client = Mock()
    client.get.return_value = {"status": "ok"}
    return client

class TestSearchService:
    """Test cases for search service."""
    
    def test_search_returns_results(self, mock_api_client):
        """Test that search returns expected results."""
        # Arrange
        service = SearchService(mock_api_client)
        query = "test query"
        
        # Act
        results = service.search(query)
        
        # Assert
        assert len(results) > 0
        assert results[0]["query"] == query
        mock_api_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_search(self):
        """Test async search functionality."""
        # Test implementation
        pass
```

**JavaScript Tests (Jest):**
```javascript
import { render, screen, fireEvent } from '@testing-library/react';
import { SearchBox } from '../SearchBox';

describe('SearchBox', () => {
  it('renders search input', () => {
    render(<SearchBox onSearch={jest.fn()} />);
    
    const input = screen.getByPlaceholderText('Search...');
    expect(input).toBeInTheDocument();
  });
  
  it('calls onSearch when form is submitted', () => {
    const mockOnSearch = jest.fn();
    render(<SearchBox onSearch={mockOnSearch} />);
    
    const input = screen.getByPlaceholderText('Search...');
    const form = screen.getByRole('form');
    
    fireEvent.change(input, { target: { value: 'test query' } });
    fireEvent.submit(form);
    
    expect(mockOnSearch).toHaveBeenCalledWith('test query');
  });
});
```

### Test Coverage

- Aim for 80%+ test coverage
- Focus on critical paths
- Test edge cases
- Include error scenarios

Check coverage:
```bash
# Python
pytest --cov=src --cov-report=html

# JavaScript
npm run test:coverage
```

## Documentation

### Code Documentation

**Python Docstrings:**
```python
def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts.
    
    Uses cosine similarity of sentence embeddings to determine
    how similar two pieces of text are semantically.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Similarity score between 0.0 (different) and 1.0 (identical)
        
    Example:
        >>> calculate_similarity("Hello world", "Hi world")
        0.85
    """
```

**TypeScript JSDoc:**
```typescript
/**
 * Formats a search result for display
 * 
 * @param result - Raw search result from API
 * @param options - Formatting options
 * @returns Formatted result ready for display
 * 
 * @example
 * ```typescript
 * const formatted = formatSearchResult(result, {
 *   maxLength: 200,
 *   highlightTerms: ['docker', 'kubernetes']
 * });
 * ```
 */
export function formatSearchResult(
  result: SearchResult,
  options?: FormatOptions
): FormattedResult {
  // Implementation
}
```

### API Documentation

- Update OpenAPI specs for API changes
- Include request/response examples
- Document error responses
- Add authentication details

### User Documentation

- Update user guide for new features
- Add screenshots for UI changes
- Create tutorials for complex features
- Update FAQ for common questions

## Pull Request Process

### 1. Before Submitting

- [ ] Run all tests locally
- [ ] Update documentation
- [ ] Add/update tests
- [ ] Run linters and formatters
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

### 2. Create Pull Request

1. Push your branch to GitHub
2. Go to the repository
3. Click "New Pull Request"
4. Select your branch
5. Fill out the PR template

**PR Title Format:**
```
feat(api): add pagination to search endpoint
fix(ui): resolve search box focus issue
docs: update installation guide for Windows
```

### 3. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes

## Screenshots (if applicable)
Add screenshots for UI changes

## Related Issues
Closes #123
Relates to #456
```

### 4. Review Process

- Maintainers will review your PR
- Address feedback promptly
- Make requested changes
- Keep PR focused and small
- Be patient and respectful

### 5. After Merge

- Delete your feature branch
- Update your local main branch
- Celebrate your contribution! 🎉

## Release Process

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Steps

1. Update version in `package.json` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Deploy to production
7. Announce release

## Getting Help

- **Discord**: Join our community server
- **GitHub Discussions**: Ask questions
- **Issue Tracker**: Report bugs
- **Email**: dev@knowledgehub.dev

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- GitHub contributors page
- Release notes
- Project website

Thank you for contributing to KnowledgeHub! Your efforts help make knowledge management better for everyone. 🙏