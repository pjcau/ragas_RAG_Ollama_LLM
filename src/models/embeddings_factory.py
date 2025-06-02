class EmbeddingsFactory:
    """Factory for creating and configuring embeddings used in the evaluation process."""
    
    @staticmethod
    def create_robust_embeddings(model_name="nomic-embed-text", num_threads=2, num_ctx=2048, timeout=30):
        """Creates robust embeddings with specified parameters."""
        return OllamaEmbeddings(
            model=model_name,
            num_thread=num_threads,
            num_ctx=num_ctx,
            timeout=timeout,
        )