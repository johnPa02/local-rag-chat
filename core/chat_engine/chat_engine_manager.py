from llama_index.core.chat_engine import SimpleChatEngine, CondensePlusContextChatEngine, CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer


class ChatEngineManager:
    def __init__(
            self,
            llm,
            retriever,
            chat_mode='condense_plus_context',
            memory_limit=3900
    ):
        self.retriever = retriever
        self.llm = llm
        self.chat_mode = chat_mode
        self.memory = ChatMemoryBuffer(token_limit=memory_limit)

    def get_engine(self):
        if self.chat_mode == 'simple':
            return SimpleChatEngine.from_defaults(llm=self.llm)
        elif self.chat_mode == 'condense_plus_context':
            return CondensePlusContextChatEngine.from_defaults(
                self.retriever,
                llm=self.llm,
                memory=self.memory
            )
        else:
            raise ValueError(f"Unsupported chat mode: {self.chat_mode}")