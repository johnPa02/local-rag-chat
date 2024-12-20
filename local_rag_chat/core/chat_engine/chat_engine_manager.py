from llama_index.core.chat_engine import SimpleChatEngine, CondensePlusContextChatEngine, CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.retrievers import RouterRetriever

class ChatEngineManager:
    def __init__(
            self,
            llm,
            retrievers,
            chat_mode='condense_plus_context',
            memory_limit=3900
    ):
        self.retrievers = retrievers
        self.llm = llm
        self.chat_mode = chat_mode
        self.memory = ChatMemoryBuffer(token_limit=memory_limit)

    def get_router_retriever(self):
        hybrid_retriever, list_retriever = self.retrievers
        list_tool = RetrieverTool.from_defaults(
            retriever=list_retriever,
            description=(
                "Useful for summarization questions related to the document. "
                " Don't use if the question only requires more specific context."
            ),
        )
        hybrid_tool = RetrieverTool.from_defaults(
            retriever=hybrid_retriever,
            description=(
                "Useful when needing to retrieve specific contexts to answer questions"
            ),
        )
        return RouterRetriever(
            selector=LLMSingleSelector.from_defaults(llm=self.llm),
            retriever_tools=[
                list_tool,
                hybrid_tool,
            ],
        )

    def get_engine(self):
        # use router retriever if there are multiple retrievers
        if len(self.retrievers) > 1:
            retriever = self.get_router_retriever()
        else:
            retriever = self.retrievers[0]

        return CondensePlusContextChatEngine.from_defaults(
            retriever,
            llm=self.llm,
            memory=self.memory
        )