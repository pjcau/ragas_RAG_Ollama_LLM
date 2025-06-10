
# OpenAI Configuration
OPENAI_API_KEY = ".."

# LLM Configuration
DEFAULT_LLM_MODEL = "deepseek-r1"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"

LLM_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.1,
    "num_predict": 100,
    "format": "json",
}

# Retry Configuration
RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1,  # secondi
}

# Dataset Configuration
DATASET_CONFIG = {
    "min_context_length": 20,
    "max_contexts": 5,
    "optimal_answer_length": 150,
}

# Custom Metrics Configuration
CUSTOM_METRICS_CONFIG = {
    "optimal_length": 150,
    "technical_word_min_length": 6,
    "answer_completeness_cap": 200,
}
