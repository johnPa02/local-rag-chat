from local_rag_chat.core.llms.base import BaseLLMModel
from llama_index.llms.ollama import Ollama
import requests

from local_rag_chat.logs.logging_config import logger
from settings import OLLAMA_BASE_URL


class OllamaModel(BaseLLMModel):
    def __init__(self, model: str):
        self.model = model
        super().__init__()

    def load_llm(self):
        if not self.model_exists():
            logger.info(f"Model {self.model} does not exist. Pulling model...")
            self.pull_model()

        self.llm = Ollama(
            model=self.model,
            base_url=OLLAMA_BASE_URL,
            request_timeout=120
        )
        logger.info(f"Loaded Ollama model: {self.model}")

    def model_exists(self):
        data = requests.get(f"{OLLAMA_BASE_URL}/api/tags").json()
        return any(item['model']==self.model for item in data['models'])

    def predict(self, query):
        response = self.llm.complete(query)
        return response

    def pull_model(self):
        payload = {
            "name": self.model
        }
        return requests.post(f"{OLLAMA_BASE_URL}/api/pull", json=payload)



