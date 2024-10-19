class EmbeddingManager:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.model = model

    def load(self):
        if self.model == "text-embedding-ada-002":
            from llama_index.embeddings.openai import OpenAIEmbedding
            return OpenAIEmbedding()
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=self.model)