from typing import Optional

from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.schema import BaseNode, logger

from core.chat_engine.chat_engine_manager import ChatEngineManager
from core.llms.ollama import OllamaModel
from core.llms.openai import OpenAIModel
from core.loaders.base import BaseLoader
from core.loaders.simple_loader import SimpleLoader
from core.retrievers.hybrid_retriever import HybridRetriever
from core.embeddings.embedding_manager import EmbeddingManager

class RAGPipeline:
    def __init__(
            self,
            llm: str = "llama3.2:1b",
            retriever_name: str = "hybrid",
            embedding: str = "BAAI/bge-small-en-v1.5",
            chat_mode: str = "condense_plus_context",
            loader: str = "simple",
    ):
        self.llm = llm
        self.retriever_name = retriever_name
        self.chat_mode = chat_mode

        self.retriever: Optional[HybridRetriever] = None
        self.embed_model: BaseEmbedding = EmbeddingManager(model=embedding).get_embedding()
        Settings.embed_model = self.embed_model

        self.llm_model: Optional[OpenAIModel | OllamaModel] = None
        self._initialize_llm()

        self.chat_engine_manager: Optional[ChatEngineManager] = None
        self.chat_engine: Optional[BaseChatEngine] = None
        self.loader: BaseLoader = SimpleLoader(embed_model=self.embed_model)

    def _initialize_llm(self):
        logger.info(f"Initializing LLM: {self.llm}")
        if "gpt" in self.llm:
            llm_model = OpenAIModel(model=self.llm)
        else:
            llm_model = OllamaModel(model=self.llm)
        self.llm_model = llm_model.get_llm()
        Settings.llm = llm_model.get_llm()

    def change_llm(self, llm: str):
        self.llm = llm
        self._initialize_llm()

    def _initialize_retriever(self, nodes: list[BaseNode]):
        logger.info(f"Initializing Retriever: {self.retriever_name}")
        if self.retriever_name == "hybrid":
            self.retriever = HybridRetriever(
                nodes=nodes,
                embed_model=self.embed_model,
                top_k=5
            )
        else:
            raise ValueError("Retriever not supported.")

    def _initialize_chat_engine(self):
        logger.info(f"Initializing Chat Engine: {self.chat_mode}")
        if not self.retriever:
            raise ValueError("Retriever not initialized.")

        self.chat_engine_manager = ChatEngineManager(
            self.llm_model,
            self.retriever,
            chat_mode=self.chat_mode
        )
        self.chat_engine = self.chat_engine_manager.get_engine()

    def stream(self, query: str):
        """
        Stream the chatbot
        using chat_history in chat_engine if we want to control the chat history
        :param query:
        :return:
        """
        if not self.chat_engine:
            self._initialize_chat_engine()
        return self.chat_engine.stream_chat(query)

    def process_documents(self, file_paths: list[str] | str):
        documents = []
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        for file in file_paths:
            documents.extend(self.loader.fit(file))
        self._initialize_retriever(documents)




