from typing import Optional, List

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import BaseNode, QueryBundle, NodeWithScore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.legacy.retrievers import BM25Retriever


class HybridRetriever(BaseRetriever):
    def __init__(
            self,
            nodes: list[BaseNode],
            embed_model: str = "BAAI/bge-small-en-v1.5",
            top_k: int = 5,
            vector_store: Optional[BasePydanticVectorStore] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.nodes = nodes
        self.embed_model = embed_model
        self.top_k = top_k
        self.vector_store = vector_store
        self.retriever = self._init_hybrid_retriever()

    def _load_index(self):
        if self.nodes is None:
            raise ValueError("Nodes must be provided to the retriever.")
        else:
            self._init_from_nodes()

    def _init_from_nodes(self):
        if self.vector_store is None:
            vector_index = VectorStoreIndex(self.nodes)
        else:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            vector_index = VectorStoreIndex(
                self.nodes, storage_context=storage_context
            )
        return vector_index

    def _init_hybrid_retriever(self):
        vector_index = self._load_index()
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=self.top_k, embed_model=self.embed_model
        )

        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index, similarity_top_k=self.top_k
        )
        # no gen query, no pass llm
        hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=3,
            num_queries=1,
            mode="reciprocal_rerank"
            # llm=self.llm
        )
        return hybrid_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        retrieve_nodes = self.retriever.retrieve(query_bundle)
        return retrieve_nodes
