class BaseLLMModel:
    def __init__(self):
        self.llm = None
        self.load_llm()
    def load_llm(self):
        """
        Load the language model
        """
        raise NotImplementedError

    def predict(self, query):
        """
        Generate predictions from the language model
        :param query: str: the input query
        :return: Any: the generated predictions
        """
        raise NotImplementedError

    def get_llm(self):
        """
        Get llm model that follows base class that llama index classes require
        """
        return self.llm