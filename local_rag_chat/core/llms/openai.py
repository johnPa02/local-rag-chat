import os
from typing import Optional
import openai
from local_rag_chat.core.llms.base import BaseLLMModel
from llama_index.llms.openai import OpenAI


class OpenAIModel(BaseLLMModel):
    def __init__(
            self,
            model: str = "gpt-4o",
            api_key: str = "",
            model_kwargs: Optional[dict] = None
    ):
        self.model = model
        self.api_key = api_key
        self.model_kwargs = model_kwargs
        super().__init__()

    def load_llm(self):

        if not self.api_key:
            from dotenv import load_dotenv
            load_dotenv()
            self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

        if self.model_kwargs is None:
            self.llm = OpenAI(model=self.model)
        else:
            self.llm = OpenAI(model=self.model, **self.model_kwargs)

    def predict(self, query):
        resp = self.llm.complete(query)
        return resp