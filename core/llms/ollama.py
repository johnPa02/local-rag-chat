from core.llms.base import BaseLLMModel
from llama_index.llms.ollama import Ollama

class OllamaModel(BaseLLMModel):
    def __init__(self, model: str):
        self.model = model
        super().__init__()

    def load_llm(self):
        self.llm = Ollama(model=self.model, request_timeout=60.0)

    def predict(self, query):
        response = self.llm.complete(query)
        return response