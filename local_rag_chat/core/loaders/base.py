class BaseLoader:
    """
    Base class for all loaders.
    """
    def __init__(self, chunk_size=512, chunk_overlap=100):
        self.chunk_size = 512
        self.chunk_overlap = 100

    def load(self, file):
        raise NotImplementedError()

    def split(self, data):
        raise NotImplementedError()

    def fit(self, file):
        documents = self.load(file)
        return self.split(documents)