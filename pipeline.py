from typing import Optional
from llama_index.core import Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.schema import BaseNode
from local_rag_chat.core.chat_engine.chat_engine_manager import ChatEngineManager
from local_rag_chat.core.llms.ollama import OllamaModel
from local_rag_chat.core.llms.openai import OpenAIModel
from local_rag_chat.core.loaders.base import BaseLoader
from local_rag_chat.core.loaders.simple_loader import SimpleLoader
from local_rag_chat.core.retrievers.hybrid_retriever import HybridRetriever
from local_rag_chat.core.embeddings.embedding_manager import EmbeddingManager
from llama_index.core import SummaryIndex
from local_rag_chat.logs.logging_config import logger


class RAGPipeline:
    def __init__(
            self,
            llm: str = "llama3.2:1b",
            embedding: str = "BAAI/bge-small-en-v1.5",
            chat_mode: str = "condense_plus_context"
    ):
        self.llm = llm
        self.chat_mode = chat_mode

        self.retrievers: list[BaseRetriever] = []
        self.embed_model: BaseEmbedding = EmbeddingManager(model=embedding).get_embedding()
        Settings.embed_model = self.embed_model

        self.llm_model: Optional[OpenAIModel | OllamaModel] = None
        self._initialize_llm()

        self.chat_engine_manager: Optional[ChatEngineManager] = None
        self.chat_engine: Optional[BaseChatEngine] = None
        self.loader: BaseLoader = SimpleLoader()

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

    def _initialize_retrievers(self, nodes: list[BaseNode]):
        logger.info(f"Initializing list of retrievers")
        hybrid_retriever = HybridRetriever(
            nodes=nodes,
            embed_model=self.embed_model,
            top_k=5
        )
        # summary retriever
        summary_index = SummaryIndex(nodes)
        list_retriever = summary_index.as_retriever()

        self.retrievers = [hybrid_retriever, list_retriever]


    def _initialize_chat_engine(self):
        logger.info(f"Initializing Chat Engine: {self.chat_mode}")
        if not self.retrievers:
            raise ValueError("Retriever list not initialized.")

        self.chat_engine_manager = ChatEngineManager(
            self.llm_model,
            self.retrievers,
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
        # TODO: clear chat engine memory for new documents
        if self.chat_engine:
            pass
        self._initialize_retrievers(documents)
