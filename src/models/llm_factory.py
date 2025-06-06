# Factory per LLM e embeddings con import condizionali
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
    """Factory per la creazione di LLM e embeddings"""
    
    @staticmethod
    def create_ultra_compatible_llm():
        """Crea un LLM ultra-compatibile con RAGAS"""
        if OPENAI_AVAILABLE and OPENAI_API_KEY != "..":
            return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
        elif OLLAMA_AVAILABLE:
            return ChatOllama(
                model=DEFAULT_LLM_MODEL,
                **LLM_CONFIG
            )
        else:
            raise ImportError("Nessun LLM disponibile. Installa langchain-ollama o langchain-openai")
    
    @staticmethod
    def create_openai_llm():
        """Crea un LLM OpenAI"""
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai non è installato")
        return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    
    @staticmethod
    def create_robust_embeddings():
        """Embeddings più robusti"""
        if OPENAI_AVAILABLE and OPENAI_API_KEY != "..":
            return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        elif OLLAMA_AVAILABLE:
            return OllamaEmbeddings(
                model=DEFAULT_EMBEDDING_MODEL
            )
        else:
            raise ImportError("Nessun embedding disponibile. Installa langchain-ollama o langchain-openai")
    
    @staticmethod
    def create_openai_embeddings():
        """Embeddings OpenAI"""
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai non è installato")
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)