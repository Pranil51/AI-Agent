from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from schemas import DBQueryPlan
import config
import logging
retrievallogger = logging.getLogger(__name__)
handler = logging.FileHandler('logs/retrieval.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
retrievallogger.addHandler(handler)
retrievallogger.setLevel(logging.DEBUG)

reranker_tokenizer = AutoTokenizer.from_pretrained(config.RERANK_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL_NAME)
reranker_model.eval()
def rerank_results(query: str, docs):
    
    pairs = [[query, doc.page_content] for doc in docs]
    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1,).float()
    results = [(doc, score) for doc, score in zip(docs, scores.tolist())]
    logging.debug(f"Reranker scores: {[score for doc, score in results]}")    
    reranked_results = sorted(results, key=lambda item: item[1], reverse=True)
    return reranked_results

from prompts import SystemPrompts
from langchain.messages import HumanMessage
class AdvancedRetriever:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
    def __call__(self, message: HumanMessage):
        """Retrieve relevant documents for multiple queries."""
        retrieval_strategist = self.llm.with_structured_output(DBQueryPlan)
        retrieval_strategy = retrieval_strategist.invoke([
            SystemPrompts.retriever,
            message
        ])
        all_results = []
        retrievallogger.info(f"Retrieval Queries: {[q_item.query for q_item in retrieval_strategy.queries]}")
        for q_item in retrieval_strategy.queries:
            result_docs = self.vector_store.similarity_search(q_item.query, k=15)
            result_docs = rerank_results(q_item.query, [doc for doc in result_docs])[:q_item.n_results]
            all_results.extend([doc for doc, score in result_docs])
        return all_results
          
