class BaseLLMModel:
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