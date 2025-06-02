class LLMFactory:
    """Factory class for creating and configuring language models (LLMs) used in the evaluation process."""

    @staticmethod
    def create_ultra_compatible_llm(model_name="llama3.2", temperature=0.1, num_predict=512, num_ctx=8192, num_thread=2, repeat_penalty=1.05, top_k=20, top_p=0.9, timeout=60):
        """Creates an ultra-compatible LLM for RAGAS."""
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            num_thread=num_thread,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            top_p=top_p,
            format="json",
            timeout=timeout,
        )

    @staticmethod
    def create_robust_embeddings(model_name="nomic-embed-text", num_thread=2, num_ctx=2048, timeout=30):
        """Creates robust embeddings for RAGAS."""
        return OllamaEmbeddings(
            model=model_name,
            num_thread=num_thread,
            num_ctx=num_ctx,
            timeout=timeout,
        )