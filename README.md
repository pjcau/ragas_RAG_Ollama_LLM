# RAGAS Evaluator

## Overview
The RAGAS Evaluator is a comprehensive Python framework designed for advanced evaluation of Retrieval-Augmented Generation (RAG) systems. It implements intelligent retry mechanisms, automatic dataset validation, and robust metric testing to ensure accurate performance assessments.

## 🚀 Features
- **🧠 RAGEvaluator Class**: Central evaluation engine with advanced retry logic
- **📊 Comprehensive Metrics**: Tests multiple RAG evaluation metrics with intelligent fallbacks
- **🔧 Dataset Validation**: Automatic dataset validation and error correction
- **⚡ Smart Retry System**: Intelligent retry mechanisms for failed metric evaluations
- **🛠️ Configurable Models**: Support for multiple LLM and embedding configurations
- **📈 Performance Monitoring**: Detailed timing and success rate tracking

## 📁 Project Structure
```
ragas-evaluator/
├── src/
│   ├── evaluator/
│   │   ├── __init__.py
│   │   ├── rag_evaluator.py          # Main evaluator class
│   │   ├── metrics_tester.py         # Individual metric testing
│   │   └── dataset_validator.py      # Dataset validation utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_factory.py           # LLM creation and configuration
│   │   └── embeddings_factory.py    # Embedding model management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── retry_handler.py         # Retry logic implementation
│   │   ├── config.py                # Configuration management
│   │   └── logger.py                # Logging utilities
│   └── data/
│       ├── __init__.py
│       ├── dataset_creator.py       # Test dataset creation
│       └── data_validator.py        # Data validation functions
├── tests/
│   ├── __init__.py
│   ├── test_evaluator.py           # Evaluator tests
│   ├── test_metrics.py             # Metric-specific tests
│   └── test_dataset.py             # Dataset validation tests
├── config/
│   ├── settings.yaml               # Main configuration
│   └── metrics_config.yaml         # Metric-specific settings
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package installation
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+ 
- Ollama (for local LLM support)
- Git

### Step-by-step Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ragas-evaluator
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "from src.evaluator.rag_evaluator import RAGEvaluator; print('Installation successful!')"
   ```

## 🚀 Quick Start

### Basic Usage

```python
from src.evaluator.rag_evaluator import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()

# Test all available metrics
working_metrics, failed_metrics = evaluator.test_individual_metrics_enhanced()

# Evaluate a specific query and documents
query = "What is machine learning?"
answer = "Machine learning is a subset of AI..."
contexts = ["Machine learning involves...", "AI systems can..."]

results = evaluator.evaluate_complete(query, answer, contexts)
```

### Advanced Usage

```python
# Custom configuration
evaluator = RAGEvaluator()

# Use only working metrics for faster evaluation
results = evaluator.evaluate_all_working_metrics(
    query="Your question here",
    answer="Generated answer",
    contexts=["Context 1", "Context 2"]
)

# Display comprehensive results
evaluator.display_comprehensive_results(
    ragas_results=results['ragas'],
    custom_results=results['custom']
)
```

## 📊 Available Metrics

### Core RAG Metrics
- **Faithfulness**: Measures factual consistency with source documents
- **Answer Relevancy**: Evaluates how well the answer addresses the question
- **Context Precision**: Assesses relevance of retrieved contexts
- **Context Recall**: Measures completeness of context retrieval

### Additional Quality Metrics
- **Answer Correctness**: Factual accuracy assessment
- **Answer Similarity**: Semantic similarity evaluation
- **Context Entity Recall**: Entity-based context evaluation
- **Coherence**: Text coherence and flow
- **Fluency**: Language fluency assessment
- **Conciseness**: Response brevity evaluation

### Custom Metrics
- **Jaccard Similarity**: Lexical overlap measurement
- **Context Coverage**: Context utilization assessment
- **Information Density**: Information content evaluation
- **Keyword Relevance**: Domain-specific keyword matching

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test module
pytest tests/test_evaluator.py

# Verbose output
pytest -v
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file:
```env
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_MODEL=llama3.2
DEFAULT_EMBEDDING_MODEL=nomic-embed-text
LOG_LEVEL=INFO
```

### Settings Configuration
Modify `config/settings.yaml`:
```yaml
models:
  llm:
    model_name: "llama3.2"
    temperature: 0.1
    timeout: 60
  embeddings:
    model_name: "nomic-embed-text"
    timeout: 30

evaluation:
  max_retries: 3
  retry_delay: 1
  max_workers: 2
```

## 🔧 Troubleshooting

### Common Issues

1. **Metric Evaluation Failures**
   ```python
   # Test individual metrics to identify issues
   working, failed = evaluator.test_individual_metrics_enhanced()
   print(f"Failed metrics: {list(failed.keys())}")
   ```

2. **LLM Connection Issues**
   ```bash
   # Verify Ollama is running
   curl http://localhost:11434/api/tags
   ```

3. **Dataset Validation Errors**
   ```python
   # Validate and fix dataset automatically
   fixed_dataset = evaluator.validate_and_fix_dataset(your_dataset)
   ```

### Performance Optimization

- Use `max_workers=1` for stability
- Increase timeouts for complex evaluations
- Cache working metrics to avoid re-testing
- Use simplified datasets for problematic metrics

## 📈 Performance Tips

1. **Use Metric Caching**: Test metrics once, reuse results
2. **Optimize Dataset Size**: Smaller contexts for faster evaluation
3. **Configure Timeouts**: Adjust based on your hardware
4. **Monitor Memory**: Large models require significant RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run tests: `pytest`
5. Commit changes: `git commit -am 'Add new feature'`
6. Push to branch: `git push origin feature/new-feature`
7. Submit a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -e .[dev]

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [RAGAS](https://github.com/explodinggradients/ragas) for the evaluation framework
- [LangChain](https://github.com/langchain-ai/langchain) for RAG components
- [Ollama](https://ollama.ai/) for local LLM support

## 📞 Support

For issues and questions:
- Create an [Issue](../../issues)
- Check [Documentation](../../wiki)
- Review [Examples](examples/)

---

**Built with ❤️ for the RAG evaluation community**