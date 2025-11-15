import logging
import os
import pickle
import re

import faiss
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from .config import config
from .data_source import get_messages

logger = logging.getLogger(__name__)
CACHE_PATH = "vector_store.pkl"
VECTOR_SEARCH_K = 10


class VectorStore:
    def __init__(self):
        logger.info("Initializing VectorStore...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.documents = []
        self.document_users = []
        self.user_names = []
        if os.path.exists(CACHE_PATH):
            try:
                self._load_from_cache()
                return
            except Exception as exc:
                logger.warning(
                    "Failed to load VectorStore cache (%s). Rebuilding from scratch.",
                    exc,
                )
        self._build_and_cache()

    def _load_from_cache(self):
        logger.info(f"Loading VectorStore from cache: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            cached_data = pickle.load(f)
            required_keys = {
                "documents",
                "document_users",
                "user_names",
                "index_bytes",
            }
            if not required_keys.issubset(cached_data.keys()):
                raise KeyError("VectorStore cache missing required fields")
            self.documents = cached_data["documents"]
            self.document_users = cached_data["document_users"]
            self.user_names = cached_data["user_names"]
            index_bytes = cached_data["index_bytes"]
            self.index = faiss.deserialize_index(index_bytes)
        logger.info("VectorStore loaded from cache successfully.")

    def _build_and_cache(self):
        logger.info(f"Cache not found at {CACHE_PATH}. Building VectorStore from scratch...")
        print("Building new vector store cache...")
        self.documents = self._prepare_documents()
        self.embeddings = self._embed_documents()
        self.index = self._create_faiss_index()
        
        with open(CACHE_PATH, "wb") as f:
            index_bytes = faiss.serialize_index(self.index)
            pickle.dump(
                {
                    "index_bytes": index_bytes,
                    "documents": self.documents,
                    "document_users": self.document_users,
                    "user_names": self.user_names,
                },
                f,
            )
        logger.info(f"VectorStore built and cached to {CACHE_PATH}")

    def _prepare_documents(self):
        records = get_messages()
        unique_documents = []
        unique_users = []
        seen_docs = set()
        all_users = set()

        for record in records:
            user_name = record.get("user_name")
            message = record.get("message")
            if user_name and message:
                doc = f"{user_name}: {message}"
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    unique_documents.append(doc)
                    unique_users.append(user_name)
                    all_users.add(user_name)

        self.document_users = unique_users
        self.user_names = sorted(list(all_users))
        return unique_documents

    def _embed_documents(self):
        logger.info(
            "Embedding %s documents with sentence-transformers/all-MiniLM-L6-v2...",
            len(self.documents),
        )
        embeddings = self.embedder.encode(
            self.documents,
            batch_size=128,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        logger.info("Document embedding complete.")
        return np.asarray(embeddings, dtype=np.float32)

    def _create_faiss_index(self):
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)
        return index

    def search(self, query: str, k: int = 5):
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
        )[0]
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        _, indices = self.index.search(np.array([query_embedding]), k)
        
        candidates = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                candidates.append((self.document_users[idx], self.documents[idx]))

        target_users = self._match_users(query)
        if target_users:
            ordered = sorted(candidates, key=lambda doc: doc[0] not in target_users)
        else:
            ordered = candidates

        return [doc for _, doc in ordered]

    def _match_users(self, query: str):
        normalized_query = self._normalize(query)
        matches = {
            user
            for user in self.user_names
            if self._normalize(user) in normalized_query
        }
        return matches

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("’", "'").lower())


class RAGQueryProcessor:
    def __init__(self):
        # When running on Google Cloud, authentication is handled automatically
        # by the service account. The API key is only needed for local development.
        if "K_SERVICE" in os.environ:
            # Authenticate using Application Default Credentials
            genai.configure()
        else:
            # For local development, require the API key
            if not config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file for local development.")
            genai.configure(api_key=config.GEMINI_API_KEY)

        self.model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        self.vector_store = VectorStore()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def answer_question(self, question: str) -> str:
        logger.info(f"Searching for context related to: '{question}'")
        context_messages = self.vector_store.search(
            question, k=VECTOR_SEARCH_K
        )
        
        if not context_messages:
            return "I couldn't find any relevant information to answer your question."

        reranked_messages = self._rerank(question, context_messages)
        
        top_k_messages = reranked_messages[:VECTOR_SEARCH_K]

        context = "\n".join(top_k_messages)
        
        prompt = f"""
        You are a careful assistant. Use ONLY the context below to answer the question.
        - If the context clearly and directly answers the question, respond with a single concise sentence in the third person.
        - If the context is related but does not definitively answer the question, respond exactly with:
          "I'm sorry, I couldn't find an answer to your question in the provided messages."
        - Never make up facts, reuse unrelated context, or infer an answer that is not explicitly stated.

        Context:
        {context}

        Question:
        {question}
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer from Gemini: {e}")
            return "I encountered an error while trying to generate an answer. Please try again."

    def _rerank(self, query: str, documents: list[str]) -> list[str]:
        pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(zip(scores, documents), reverse=True)
        
        return [doc for _, doc in scored_docs]
