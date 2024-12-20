from typing import Optional, List
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import BaseNode, QueryBundle, NodeWithScore
from llama_index.core.vector_stores.types import BasePydanticVectorStore


class HybridRetriever(BaseRetriever):
    def __init__(
            self,
            nodes: list[BaseNode],
            embed_model: BaseEmbedding,
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
            return self._init_from_nodes()

    def _init_from_nodes(self):
        if self.vector_store is None:
            vector_index = VectorStoreIndex(self.nodes, embed_model=self.embed_model)
        else:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            vector_index = VectorStoreIndex(
                self.nodes, storage_context=storage_context, embed_model=self.embed_model
            )
        return vector_index

    def _init_hybrid_retriever(self):
        vector_index = self._load_index()
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=self.top_k
        )

        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index, similarity_top_k=self.top_k
        )
        hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=3,
            num_queries=1,
            retriever_weights=[0.6, 0.4],
            use_async=True,
            mode=FUSION_MODES.RECIPROCAL_RANK
            # llm=self.llm
        )
        return hybrid_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        retrieve_nodes = self.retriever.retrieve(query_bundle)
        return retrieve_nodes
