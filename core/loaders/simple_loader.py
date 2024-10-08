from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import Document
from core.loaders.base import BaseLoader
import fitz

class SimpleLoader(BaseLoader):
    """
    Simple loader that handles simple file types like PDFs.
    Supports splitting by semantic or sentence.
    """
    def __init__(self, embed_model=None):
        super().__init__()
        self.embed_model = embed_model

    def load(self, file):
        reader = fitz.open(file)
        text = ""
        for page_num in range(reader.page_count):
            page = reader.load_page(page_num)
            text += page.get_text()
        return [Document(text=text)]

    def split(self, documents):
        if self.embed_model:
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
            )
        else:
            node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return node_parser.get_nodes_from_documents(documents)
