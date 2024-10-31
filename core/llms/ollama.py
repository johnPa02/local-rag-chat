from core.llms.base import BaseLLMModel
from llama_index.llms.ollama import Ollama
import requests

class OllamaModel(BaseLLMModel):
    def __init__(self, model: str):
        self.model = model
        super().__init__()

    def load_llm(self):
        self.llm = Ollama(model=self.model, request_timeout=60.0)

    def predict(self, query):
        response = self.llm.complete(query)
        return response

    def list_models(self):
        return requests.get("http://ollama_server:11434/api/tags").json()

    def pull_model(self):
        payload = {
            "name": self.model
        }
        return requests.post("http://ollama_server:11434/api/pull", json=payload).json()

    def check_model_exists(self):
        pass

