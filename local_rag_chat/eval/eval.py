import pandas as pd
import asyncio
from local_rag_chat.core.llms.ollama import OllamaModel
from local_rag_chat.core.llms.openai import OpenAIModel
from local_rag_chat.core.loaders.simple_loader import SimpleLoader
from local_rag_chat.core.embeddings.embedding_manager import EmbeddingManager
from local_rag_chat.core.retrievers.hybrid_retriever import HybridRetriever
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    FaithfulnessEvaluator
)
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core import Settings, VectorStoreIndex
import argparse
from local_rag_chat.logs.logging_config import logger

def display_results(name, eval_results)-> pd.DataFrame:
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    columns = {
        "retrievers": [name],
        **{k: [full_df[k].mean()] for k in ["hit_rate", "mrr"]},
    }

    metric_df = pd.DataFrame(columns)

    return metric_df


class RagEvalPipeline:
    def __init__(self,
                 dataset_path: str,
                 qa_dataset_path: str,
                 eval_model: str = "gpt-4o",
                 llm: str = "llama3.2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 100,
                 top_k: int = 5
    ):
        self.top_k = top_k
        # dataset
        loader = SimpleLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.nodes = loader.fit(dataset_path)
        self.qa_dataset = EmbeddingQAFinetuneDataset.from_json(qa_dataset_path)

        # retriever
        self.embed_model = EmbeddingManager(model='BAAI/bge-small-en-v1.5').get_embedding()
        Settings.embed_model = self.embed_model

        # llm
        self.gpt4 = OpenAIModel(model=eval_model).get_llm()
        self.llm = OllamaModel(model=llm).get_llm()
        Settings.llm = self.llm

    async def retrieval_evaluator(self) -> pd.DataFrame:
        hybrid_retriever = HybridRetriever(
            nodes=self.nodes,
            embed_model=self.embed_model,
            top_k=self.top_k)
        retriever_evaluator =  RetrieverEvaluator.from_metric_names(
            ["hit_rate", "mrr"], retriever=hybrid_retriever
        )
        eval_results = await retriever_evaluator.aevaluate_dataset(self.qa_dataset)
        return display_results("hybrid retriever", eval_results)

    async def response_evaluator(self):
        """
        Evaluate the responses from the llm
        Only use FaithfulnessEvaluator for now
        :return:
        """
        vector_index = VectorStoreIndex(self.nodes)
        query_engine = vector_index.as_query_engine()
        queries = list(self.qa_dataset.queries.values())

        coroutines = [query_engine.aquery(q) for q in queries]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        print("finished query")

        evaluator = FaithfulnessEvaluator(llm=self.gpt4)
        total_correct = sum(1 if evaluator.evaluate_response(response=r).passing else 0 for r in results)
        return total_correct, len(results)

    def generate_qa_dataset(self, output_path: str) -> None:
        qa_dataset = generate_question_context_pairs(
            self.nodes,
            llm = self.gpt4,
            num_questions_per_chunk=2
        )
        qa_dataset.save_json(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_type",
        type=str,
        default="both",
        choices=["retrieval", "response", "both"],
        help="Type of evaluation to perform (retrieval, response, qa_dataset)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='./data/paul_graham/paul_graham_essays.txt',
        help="Path to the dataset file (.txt, .pdf,...)"
    )
    parser.add_argument(
        "--qa_dataset_path",
        type=str,
        default='./data/paul_graham/paul_graham.json',
        help="Path to the question-context pair dataset"
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt-4o",
        help="Model used for evaluation"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="llama3.2",
        help="Model used for generation"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size for splitting text"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=100,
        help="Chunk overlap for splitting text"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top k retrievals"
    )

    async def main(args):
        rag_eval = RagEvalPipeline(
            dataset_path=args.dataset_path,
            qa_dataset_path=args.qa_dataset_path,
            eval_model=args.eval_model,
            llm=args.llm,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        logger.info("Starting evaluation")
        if args.eval_type != "response":
            results = await rag_eval.retrieval_evaluator()
            print('Retrieval Evaluation:')
            print(results)
            results.to_csv(
                f"retrieval_{args.chunk_size}_{args.chunk_overlap}_{args.top_k}.csv",
                index=False
            )

        if args.eval_type != "retrieval":
            correct, total = await rag_eval.response_evaluator()
            print('Response Evaluation (Faithfulness):')
            print(f"Correct: {correct}/{total}")
            # save results
            with open(f"response_{args.llm}_{args.chunk_size}_{args.chunk_overlap}.txt", "w") as f:
                f.write(f"Correct: {correct}/{total}")

    args = parser.parse_args()
    asyncio.run(main(args))
