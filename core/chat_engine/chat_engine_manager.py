from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer


class ChatEngineManager:
    def __init__(self, llm, query_engine, chat_mode='simple', memory_limit=3900):
        self.query_engine = query_engine
        self.llm = llm
        self.chat_mode = chat_mode
        self.memory =  ChatMemoryBuffer(tokens_limit=memory_limit)

    def get_engine(self):
        if self.chat_mode == 'simple':
            return SimpleChatEngine.from_defaults(llm=self.llm)
        else:
            raise ValueError(f"Unsupported chat mode: {self.chat_mode}")