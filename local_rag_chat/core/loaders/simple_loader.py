from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.core.schema import BaseNode

from local_rag_chat.core.loaders.base import BaseLoader
import fitz
import re
from typing import List

class SimpleLoader(BaseLoader):
    """
    Simple loader that handles simple file types like PDFs.
    Supports splitting by semantic or sentence.
    """
    def __init__(self,
                embed_model=None,
                chunk_size=512,
                chunk_overlap=100
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embed_model = embed_model

    @staticmethod
    def _filter_text(text: str) -> str:
        """
        Filter text to remove unnecessary whitespaces and line breaks.
        """
        # Remove leading and trailing whitespaces
        text = re.sub(r"\s+\n\s+", "\n", text)
        # Remove spaces before line breaks
        text = re.sub(r"\s+\n", "\n", text)
        # Remove spaces after line breaks
        text = re.sub(r"\n\s+", "\n", text)
        # Remove line breaks which are not at the end of a sentence
        text = re.sub(r"(?<![.!?])\n", " ", text)
        # Limit the number of consecutive line breaks to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def load(self, file: str)-> List[Document]:
        reader = fitz.open(file)
        text = []
        for page_num in range(reader.page_count):
            page = reader.load_page(page_num)
            page_text = page.get_text()
            page_text = self._filter_text(page_text)
            text.append(page_text)
        text = " ".join(text)
        return [Document(text=text)]

    def split(self, documents: List[Document]) -> List[BaseNode]:
        if self.embed_model:
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
            )
        else:
            node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return node_parser.get_nodes_from_documents(documents)
