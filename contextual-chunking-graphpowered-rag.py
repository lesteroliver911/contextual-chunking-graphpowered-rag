import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
import cohere
import numpy as np
import heapq
import logging
import time
from llama_parse import LlamaParse
from anthropic import Anthropic
from sklearn.metrics import ndcg_score
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')

class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")

class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="Whether the current context provides a complete answer to the query")
    answer: str = Field(description="The current answer based on the context, if any")

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=1000)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None)
        self.anthropic_client = Anthropic()

    def process_documents(self, documents: List[str]) -> Tuple[List[Document], FAISS]:
        chunks = self.text_splitter.create_documents(documents)
        contextualized_chunks = self._generate_contextualized_chunks(documents[0], chunks)
        vector_store = FAISS.from_documents(contextualized_chunks, self.embeddings)
        return chunks, vector_store

    def _generate_contextualized_chunks(self, document: str, chunks: List[Document]) -> List[Document]:
        contextualized_chunks = []
        for i, chunk in enumerate(chunks):
            context = self._generate_context(document, chunk.page_content, i, len(chunks))
            contextualized_content = f"{chunk.page_content}\n\nContext: {context}"
            contextualized_chunks.append(Document(page_content=contextualized_content, metadata=chunk.metadata))
        return contextualized_chunks

    def _generate_context(self, document: str, chunk: str, chunk_index: int, total_chunks: int) -> str:
        response = self.anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>{document}</document>",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": f"Generate context for chunk {chunk_index+1} out of {total_chunks}: <chunk>{chunk}</chunk>"
                        },
                    ]
                },
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        return response.content[0].text

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.concept_cache = {}
        self.edges_threshold = 0.8

    def build_graph(self, splits: List[Document], llm: ChatOpenAI, embedding_model: OpenAIEmbeddings):
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        logging.info(f"Embeddings shape: {embeddings.shape}")
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits: List[Document]):
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits: List[Document], embedding_model: OpenAIEmbeddings) -> np.ndarray:
        texts = [split.page_content for split in splits]
        embeddings = embedding_model.embed_documents(texts)
        return np.array(embeddings)  # Convert to numpy array before returning

    def _extract_concepts(self, splits: List[Document], llm: ChatOpenAI):
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        
        for i, split in enumerate(splits):
            if split.page_content not in self.concept_cache:
                concepts = concept_chain.invoke({"text": split.page_content}).concepts_list
                self.concept_cache[split.page_content] = concepts
            self.graph.nodes[i]['concepts'] = self.concept_cache[split.page_content]

    def _add_edges(self, embeddings: np.ndarray):
        similarity_matrix = np.dot(embeddings, embeddings.T)
        num_nodes = len(self.graph.nodes)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                similarity_score = similarity_matrix[i][j]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[i]['concepts']) & set(self.graph.nodes[j]['concepts'])
                    edge_weight = self._calculate_edge_weight(i, j, similarity_score, shared_concepts)
                    self.graph.add_edge(i, j, weight=edge_weight, similarity=similarity_score, shared_concepts=list(shared_concepts))

    def _calculate_edge_weight(self, node1: int, node2: int, similarity_score: float, shared_concepts: set, alpha: float = 0.7, beta: float = 0.3) -> float:
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

class QueryEngine:
    def __init__(self, vector_store: FAISS, knowledge_graph: KnowledgeGraph, llm: ChatOpenAI):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_context_length = 4000
        self.answer_check_chain = self._create_answer_check_chain()
        self.chunks = [doc.page_content for doc in vector_store.docstore._dict.values()]  # Add this line
        self.bm25 = self._create_bm25_index()
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

    def _create_answer_check_chain(self):
        answer_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the query: '{query}'\n\nAnd the current context:\n{context}\n\nDoes this context provide a complete answer to the query? If yes, provide the answer. If no, state that the answer is incomplete.\n\nIs complete answer (Yes/No):\nAnswer (if complete):"
        )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck)

    def _create_bm25_index(self):
        tokenized_chunks = [chunk.split() for chunk in self.chunks]  # Use self.chunks here
        return BM25Okapi(tokenized_chunks)

    def _hybrid_search(self, query: str, k: int = 20) -> List[Document]:
        logging.info(f"Performing hybrid search for query: {query}")
        semantic_results = self.vector_store.similarity_search_with_score(query, k=k)
        logging.info(f"Semantic search returned {len(semantic_results)} results")
        
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]
        logging.info(f"BM25 top indices: {bm25_top_indices}")
        
        content_to_doc = {doc.page_content: doc for doc in self.vector_store.docstore._dict.values()}
        logging.info(f"Number of documents in vector store: {len(content_to_doc)}")
        
        bm25_results = []
        for i in bm25_top_indices:
            if i < len(self.chunks):
                content = self.chunks[i]
                if content in content_to_doc:
                    doc = content_to_doc[content]
                    bm25_results.append((doc, bm25_scores[i]))
                else:
                    logging.warning(f"Content for index {i} not found in vector store")
            else:
                logging.warning(f"Index {i} is out of bounds for self.chunks (length: {len(self.chunks)})")

        logging.info(f"BM25 search returned {len(bm25_results)} results")

        combined_results = semantic_results + bm25_results
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in combined_results[:k]]

    def _rerank_results(self, query: str, documents: List[Document], k: int = 3) -> List[Document]:
        doc_contents = [doc.page_content for doc in documents]
        reranked = self.cohere_client.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=doc_contents,
            top_n=k
        )
        return [documents[result.index] for result in reranked.results]

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete, response.answer

    def _expand_context(self, query: str, relevant_docs: List[Document]) -> Tuple[str, List[int], Dict[int, str], str]:
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}

        for doc in relevant_docs:
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if
                                self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)
            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        while priority_queue:
            current_priority, current_node = heapq.heappop(priority_queue)
            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                node_concepts_set = set(node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']
                        distance = current_priority + (1 / edge_weight)

                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

        if not final_answer:
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
            )
            response_chain = response_prompt | self.llm
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        with get_openai_callback() as cb:
            relevant_docs = self._hybrid_search(query)
            relevant_docs = self._rerank_results(query, relevant_docs)
            expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)

            print(f"\nTotal Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")

        return final_answer, traversal_path, filtered_content

    def evaluate_retrieval(self, queries: List[Dict[str, Any]], k: int = 20):
        results = []
        for query_item in queries:
            query = query_item['query']
            golden_chunk_uuids = query_item['golden_chunk_uuids']
            
            retrieved_docs = self._hybrid_search(query, k=k)
            retrieved_docs = self._rerank_results(query, retrieved_docs, k=k)
            
            retrieved_contents = [doc.page_content for doc in retrieved_docs]
            relevant_docs = [doc for doc in self.vector_store.docstore._dict.values() if doc.metadata.get('uuid') in golden_chunk_uuids]
            
            hit = any(doc in relevant_docs for doc in retrieved_docs)
            reciprocal_rank = next((1 / (rank + 1) for rank, doc in enumerate(retrieved_docs) if doc in relevant_docs), 0)
            precision = len(set(retrieved_docs) & set(relevant_docs)) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs) if relevant_docs else 0
            
            relevance_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
            ideal_scores = [1] * len(relevant_docs) + [0] * (len(retrieved_docs) - len(relevant_docs))
            ndcg = ndcg_score([ideal_scores], [relevance_scores]) if relevance_scores else 0
            
            results.append({
                "hit_rate": int(hit),
                "mrr": reciprocal_rank,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg
            })
        
        return results

class Visualizer:
    @staticmethod
    def visualize_traversal(graph: nx.Graph, traversal_path: List[int]):
        traversal_graph = nx.DiGraph()
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)

        fig, ax = plt.subplots(figsize=(16, 12))
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        edges = traversal_graph.edges()
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos, edgelist=edges, edge_color=edge_weights, edge_cmap=plt.cm.Blues, width=2, ax=ax)

        nx.draw_networkx_nodes(traversal_graph, pos, node_color='lightblue', node_size=3000, ax=ax)

        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start, end = traversal_path[i], traversal_path[i + 1]
            start_pos, end_pos = pos[start], pos[end]
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)
            arrow = patches.FancyArrowPatch(start_pos, end_pos, connectionstyle=f"arc3,rad={0.3}", color='red',
                                            arrowstyle="->", mutation_scale=20, linestyle='--', linewidth=2, zorder=4)
            ax.add_patch(arrow)

        labels = {}
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label

        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

        start_node, end_node = traversal_path[0], traversal_path[-1]
        nx.draw_networkx_nodes(traversal_graph, pos, nodelist=[start_node], node_color='lightgreen', node_size=3000, ax=ax)
        nx.draw_networkx_nodes(traversal_graph, pos, nodelist=[end_node], node_color='lightcoral', node_size=3000, ax=ax)

        ax.set_title("Graph Traversal Flow")
        ax.axis('off')

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Edge Weight', rotation=270, labelpad=15)

        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Regular Edge')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Traversal Path')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Start Node')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15, label='End Node')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path: List[int], filtered_content: Dict[int, str]):
        print("\nFiltered content of visited nodes in order of traversal:")
        for i, node in enumerate(traversal_path):
            print(f"\nStep {i + 1} - Node {node}:")
            print(f"Filtered Content: {filtered_content.get(node, 'No filtered content available')[:200]}...")
            print("-" * 50)

class GraphRAG:
    def __init__(self, documents: List[str]):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.embedding_model = OpenAIEmbeddings()
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None
        self.visualizer = Visualizer()
        self.process_documents(documents)

    def process_documents(self, documents: List[str]):
        all_splits = []
        for doc in documents:
            splits = self.document_processor.text_splitter.create_documents([doc])
            all_splits.extend(splits)
        vector_store = FAISS.from_documents(all_splits, self.embedding_model)
        self.knowledge_graph.build_graph(all_splits, self.llm, self.embedding_model)
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.llm)

    def query(self, query: str) -> str:
        response, traversal_path, filtered_content = self.query_engine.query(query)
        if traversal_path:
            self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
            self.visualizer.print_filtered_content(traversal_path, filtered_content)
        else:
            print("No traversal path to visualize.")
        return response

    def evaluate(self, queries: List[Dict[str, Any]], k: int = 20):
        return self.query_engine.evaluate_retrieval(queries, k)

def load_pdf_with_llama_parse(pdf_path: str) -> str:
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables.")
    parser = LlamaParse(result_type="markdown", api_key=api_key)
    try:
        documents = parser.load_data(pdf_path)
        if not documents:
            raise ValueError("No content extracted from the PDF.")
        return " ".join([doc.text for doc in documents])
    except Exception as e:
        logging.error(f"Error while parsing the file '{pdf_path}': {str(e)}")
        raise

def main():
    # Load the PDF document
    pdf_path = input("Enter the path to your PDF file: ")
    try:
        document = load_pdf_with_llama_parse(pdf_path)
    except Exception as e:
        logging.error(f"Failed to load or parse the PDF: {str(e)}")
        return

    # Initialize and process the document
    graph_rag = GraphRAG([document])

    # Query loop
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        response = graph_rag.query(query)
        print(f"\nResponse: {response}")

    # Optionally, run evaluation if you have a test set
    # test_queries = [...]  # Load your test queries here
    # evaluation_results = graph_rag.evaluate(test_queries)
    # print("Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    main()
