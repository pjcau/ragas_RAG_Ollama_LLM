# Factory for LLM and embeddings with conditional imports
try:
    from langchain_ollama import ChatOllama
    from langchain_ollama.embeddings import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ChatOllama = None
    OllamaEmbeddings = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None
    OpenAIEmbeddings = None

from ..config.settings import OPENAI_API_KEY, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, LLM_CONFIG


class LLMFactory:
    """Factory for creating LLMs and embeddings"""

    @staticmethod
    def create_ultra_compatible_llm():
        """Creates an ultra-compatible LLM with RAGAS"""
        if OPENAI_AVAILABLE and OPENAI_API_KEY != "..":
            return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
        elif OLLAMA_AVAILABLE:
            return ChatOllama(
                model=DEFAULT_LLM_MODEL,
                **LLM_CONFIG
            )
        else:
            raise ImportError(
                "No LLM available. Install langchain-ollama or langchain-openai")

    @staticmethod
    def create_openai_llm():
        """Creates an OpenAI LLM"""
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai is not installed")
        return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

    @staticmethod
    def create_robust_embeddings():
        """More robust embeddings"""
        if OPENAI_AVAILABLE and OPENAI_API_KEY != "..":
            return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        elif OLLAMA_AVAILABLE:
            return OllamaEmbeddings(
                model=DEFAULT_EMBEDDING_MODEL
            )
        else:
            raise ImportError(
                "No embeddings available. Install langchain-ollama or langchain-openai")

    @staticmethod
    def create_openai_embeddings():
        """OpenAI embeddings"""
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai is not installed")
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
