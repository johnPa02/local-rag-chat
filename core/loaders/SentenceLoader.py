from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

from core.loaders.base import BaseLoader
import fitz

class SentenceLoader(BaseLoader):
    """
    Loader for sentence data.
    """
    def __init__(self):
        super().__init__()

    def load(self, file):
        reader = fitz.open(file)
        text = ""
        for page_num in range(reader.page_count):
            page = reader.load_page(page_num)
            text += page.get_text()
        return [Document(text=text)]

    def split(self, documents):
        node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return node_parser.get_nodes_from_documents(documents)
