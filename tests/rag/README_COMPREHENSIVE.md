# RAG System Testing Framework

Comprehensive Test-Driven Development framework for KnowledgeHub RAG system, created by Peter Verschuere, TDD specialist.

## 🎯 Overview

This testing framework provides complete coverage for the KnowledgeHub RAG system including:

- **Advanced RAG Pipeline** with 6 chunking strategies and 6 retrieval strategies
- **GraphRAG Service** with Neo4j knowledge graph integration
- **LlamaIndex Integration** with mathematical optimizations
- **Performance Optimization** services and intelligent caching
- **Distributed Architecture** validation across multiple services

## 🏗️ Architecture

```
tests/rag/
├── conftest_rag.py              # RAG-specific test fixtures
├── pytest.ini                  # RAG test configuration
├── test_runner.py               # Comprehensive test orchestrator
├── test_quality_gates.py        # Quality gates and CI/CD integration
├── unit/                        # Unit tests for individual components
│   ├── test_rag_chunking.py    # Chunking strategies (6 strategies)
│   ├── test_rag_retrieval.py   # Retrieval strategies (6 strategies)
│   └── test_rag_pipeline.py    # Complete pipeline integration
├── integration/                 # End-to-end integration tests
│   └── test_rag_e2e.py         # Full workflow validation
├── performance/                 # Performance and load testing
│   └── test_rag_performance.py # Benchmarks and stress testing
├── fixtures/                    # Test data and utilities
└── reports/                     # Generated test reports
```

## 🚀 Quick Start

### Run All Tests
```bash
cd /opt/projects/knowledgehub/tests/rag
python test_runner.py --suite all --verbose
```

### Run Specific Test Suites
```bash
# Unit tests only (fast)
python test_runner.py --suite unit

# Integration tests
python test_runner.py --suite integration

# Performance tests
python test_runner.py --suite performance

# Quality gates validation
python test_runner.py --suite quality
```

### Run with Pytest Directly
```bash
# All RAG tests
pytest -m rag

# Specific test categories
pytest -m "unit and rag"
pytest -m "performance and rag" 
pytest -m "integration and rag"

# Specific components
pytest -m chunking
pytest -m retrieval
pytest -m graphrag
pytest -m llamaindex
```

## 📊 Test Coverage

### Unit Tests (>80% coverage required)

**Chunking Strategies** (6 strategies tested):
- ✅ Semantic chunking with sentence boundaries
- ✅ Sliding window with overlap
- ✅ Recursive character splitting
- ✅ Proposition-based chunking
- ✅ Hierarchical multi-level chunking
- ✅ Adaptive context-aware sizing

**Retrieval Strategies** (6 strategies tested):
- ✅ Vector similarity search
- ✅ Hybrid vector + keyword search
- ✅ Ensemble multiple methods with voting
- ✅ Iterative progressive refinement
- ✅ Graph-enhanced retrieval
- ✅ Adaptive query-dependent strategy

**Pipeline Components**:
- ✅ Document ingestion and storage
- ✅ Query preprocessing (HyDE)
- ✅ Context construction and length limits
- ✅ Response generation and self-correction
- ✅ Error handling and edge cases

### Integration Tests

**End-to-End Workflows**:
- ✅ Complete document → query → response pipeline
- ✅ Multi-service integration (PostgreSQL, Redis, Weaviate, Neo4j)
- ✅ Cross-component data flow validation
- ✅ Concurrent pipeline usage
- ✅ Multi-turn conversational RAG
- ✅ Domain-specific document retrieval
- ✅ Multilingual content handling

**Service Integration**:
- ✅ Database operations and consistency
- ✅ Caching integration and performance
- ✅ Error recovery across components
- ✅ Metadata flow through pipeline

### Performance Tests

**Benchmarks** (thresholds enforced):
- ✅ Chunking: <100ms for 10KB documents
- ✅ Retrieval: <200ms for top-10 results  
- ✅ End-to-end: <500ms for simple queries
- ✅ Memory usage: <1GB for standard test suite

**Load Testing**:
- ✅ Concurrent user simulation (5-50 users)
- ✅ Memory leak detection (100+ operations)
- ✅ Stress testing with large documents (500KB+)
- ✅ Database performance under heavy load
- ✅ Throughput measurement and validation

**Performance Regression**:
- ✅ Baseline establishment and monitoring
- ✅ Automated regression detection
- ✅ Performance trend analysis

## 🔧 Configuration

### Environment Variables
```bash
# Test database
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/knowledgehub_test

# Service endpoints
REDIS_URL=redis://localhost:6381
WEAVIATE_URL=http://localhost:8090
NEO4J_URL=bolt://localhost:7687

# Test configuration
RAG_TEST_TIMEOUT=300
RAG_PERFORMANCE_BASELINE_PATH=/tmp/rag_baselines.json
```

### Quality Gates Thresholds
```python
QUALITY_GATES = {
    "performance_thresholds": {
        "chunking_time_ms": 100,
        "retrieval_time_ms": 200,
        "end_to_end_time_ms": 500,
        "memory_usage_mb": 1000
    },
    "accuracy_thresholds": {
        "retrieval_precision": 0.8,
        "retrieval_recall": 0.7,
        "answer_relevance": 0.75
    },
    "coverage_requirements": {
        "code_coverage": 80,
        "critical_path_coverage": 90
    }
}
```

## 🧪 Test Categories

### Markers and Filtering

| Marker | Description | Example Usage |
|--------|-------------|---------------|
| `unit` | Unit tests for individual components | `pytest -m unit` |
| `integration` | Integration tests across modules | `pytest -m integration` |
| `performance` | Performance and benchmark tests | `pytest -m performance` |
| `load` | Load testing under stress | `pytest -m load` |
| `quality_gates` | Quality gates validation | `pytest -m quality_gates` |
| `slow` | Long-running tests | `pytest -m "not slow"` |
| `chunking` | Chunking strategy tests | `pytest -m chunking` |
| `retrieval` | Retrieval strategy tests | `pytest -m retrieval` |
| `graphrag` | GraphRAG Neo4j tests | `pytest -m graphrag` |
| `llamaindex` | LlamaIndex integration | `pytest -m llamaindex` |

### Test Selection Examples
```bash
# Fast tests only (exclude slow)
pytest -m "rag and not slow"

# Performance tests for specific components  
pytest -m "performance and chunking"

# Integration tests excluding load tests
pytest -m "integration and not load"

# Critical path tests only
pytest -m "unit and (chunking or retrieval)"
```

## 📈 Performance Monitoring

### Benchmark Results Format
```json
{
  "chunking_performance": {
    "sliding_window": {
      "1000": {"time_ms": 15, "chunks": 4, "throughput_kb_s": 66.7},
      "10000": {"time_ms": 85, "chunks": 25, "throughput_kb_s": 117.6}
    }
  },
  "retrieval_performance": {
    "vector_search": {
      "10_results": {"time_ms": 120, "precision": 0.85}
    }
  },
  "e2e_performance": {
    "simple_query": {"time_ms": 350, "chunks_used": 3},
    "complex_query": {"time_ms": 480, "chunks_used": 8}
  }
}
```

### Memory Profiling
- Automatic memory leak detection
- Memory usage tracking across operations
- Performance regression alerts
- Resource utilization monitoring

## 🔍 Quality Gates

### Automated Validation
- ✅ Code coverage >80% (>90% for critical paths)
- ✅ Performance benchmarks within thresholds
- ✅ Memory usage limits enforced
- ✅ Security vulnerability scanning
- ✅ API response time validation
- ✅ Integration health checks
- ✅ Data quality metrics

### CI/CD Integration
- ✅ Build pipeline validation
- ✅ Deployment readiness checks
- ✅ Environment-specific validation
- ✅ Automated testing pipeline
- ✅ Performance monitoring setup
- ✅ Alerting configuration

## 🚨 Troubleshooting

### Common Issues

**Test Database Connection**:
```bash
# Check PostgreSQL test database
psql -h localhost -p 5433 -U postgres -d knowledgehub_test

# Reset test database
dropdb -h localhost -p 5433 -U postgres knowledgehub_test
createdb -h localhost -p 5433 -U postgres knowledgehub_test
```

**Missing Dependencies**:
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov
pip install nltk spacy  # For semantic chunking
pip install neo4j weaviate-client redis  # For service integration
```

**Performance Test Failures**:
- Check system resources (CPU, memory)
- Verify no other processes competing for resources
- Review performance baseline values
- Consider environment-specific adjustments

### Debug Mode
```bash
# Verbose output with debugging
pytest -v -s --tb=long -m rag

# Single test with full output
pytest -v -s tests/rag/unit/test_rag_chunking.py::TestAdvancedChunker::test_semantic_chunking

# Performance profiling
pytest -v -s --durations=0 -m performance
```

## 📊 Reporting

### Generated Reports

**Test Results**: `/opt/projects/knowledgehub/tests/rag/reports/`
- `rag_test_results.json` - Machine-readable results
- `rag_test_report.md` - Human-readable summary
- `coverage-unit.xml` - Coverage data (JUnit format)
- `htmlcov/` - HTML coverage report

**Performance Reports**:
- Benchmark comparisons with baselines
- Performance trend analysis
- Memory usage profiling
- Throughput measurements

### Example Report Output
```
# RAG System Test Report

**Status**: ✅ PASSED
**Generated**: 2025-08-07 14:30:15
**Duration**: 145.67 seconds

## Test Suite Results

### Unit Tests
- **Status**: ✅ PASSED  
- **Duration**: 45.23 seconds
- **Coverage**: 84.8% (above 80% threshold)

### Integration Tests  
- **Status**: ✅ PASSED
- **Duration**: 67.89 seconds
- **E2E Workflows**: All validated

### Performance Tests
- **Status**: ✅ PASSED
- **Duration**: 32.55 seconds  
- **Benchmarks**: All within thresholds

### Quality Gates
- **Status**: ✅ PASSED
- **Security**: No vulnerabilities detected
- **Quality**: All metrics above requirements
```

## 🔄 Continuous Integration

### GitHub Actions Example
```yaml
name: RAG System Tests

on: [push, pull_request]

jobs:
  test-rag:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5433:5432
      redis:
        image: redis:6
        ports:
          - 6381:6379
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r test-requirements.txt
    
    - name: Run RAG Tests
      run: |
        cd tests/rag
        python test_runner.py --suite all --verbose
    
    - name: Upload Coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage-unit.xml
```

## 🎖️ Best Practices

### Test Organization
- **Single Responsibility**: Each test validates one specific behavior
- **Clear Naming**: Test names describe what is being tested
- **Arrange-Act-Assert**: Clear test structure
- **Independent Tests**: No dependencies between tests
- **Fast Feedback**: Unit tests complete in <5 seconds

### Mocking Strategy
- **External Services**: Always mock network calls
- **Database**: Use test database or mock for unit tests  
- **File System**: Mock file operations
- **Time-Dependent**: Mock datetime for consistent tests
- **Hardware-Dependent**: Mock GPU operations

### Performance Testing
- **Baseline Establishment**: Record initial performance metrics
- **Environment Consistency**: Run on consistent hardware
- **Resource Isolation**: Avoid interference from other processes
- **Statistical Significance**: Multiple test runs for accuracy
- **Regression Detection**: Alert on >20% performance degradation

## 📚 References

- [pytest Documentation](https://docs.pytest.org/)
- [Test-Driven Development Best Practices](https://martinfowler.com/bliki/TestDrivenDevelopment.html)
- [Python Testing 101](https://realpython.com/python-testing/)
- [RAG System Architecture](../RAG_IMPLEMENTATION_GUIDE.md)
- [KnowledgeHub API Documentation](../api/README.md)

---

**Author**: Peter Verschuere - Test-Driven Development Expert  
**Last Updated**: August 7, 2025  
**Framework Version**: 1.0.0
EOF < /dev/null
