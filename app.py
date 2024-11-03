from pages.chatbot import App
from pipeline import RAGPipeline
from logs.logging_config import logger

logger.info("Starting the app...")

pipeline = RAGPipeline(
    llm='llama3.2:1b',
    retriever_name='hybrid',
    embedding='BAAI/bge-small-en-v1.5',
    chat_mode='condense_plus_context'
)

app = App(pipeline)
demo = app.build()
demo.launch(server_name='0.0.0.0')

logger.info("App started successfully!")