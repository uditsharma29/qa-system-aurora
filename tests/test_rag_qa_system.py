import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from app import rag_qa_system


class FakeEmbedder:
    def encode(
        self,
        texts,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True,
    ):
        vectors = []
        for text in texts:
            if "Layla" in text:
                vectors.append([0.0])
            elif "Vikram" in text:
                vectors.append([5.0])
            else:
                vectors.append([10.0])
        return np.asarray(vectors, dtype=np.float32)


class RAGQASystemTests(unittest.TestCase):
    @mock.patch("os.path.exists", return_value=False)
    @mock.patch("app.rag_qa_system.get_messages")
    def test_vector_store_builds_and_searches(self, mock_get_messages, mock_exists):
        mock_get_messages.return_value = [
            {"user_name": "Layla", "message": "I'm planning a trip to London next month."},
            {"user_name": "Vikram", "message": "I need a car in New York."},
        ]

        # In-memory vector store for testing
        store = rag_qa_system.VectorStore()

        # Test document preparation
        self.assertEqual(len(store.documents), 2)
        self.assertIn("Layla: I'm planning a trip to London next month.", store.documents)

        # Test search
        results = store.search("When is the London trip?", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Layla: I'm planning a trip to London next month.")

    @mock.patch("os.path.exists", return_value=False)
    @mock.patch("app.rag_qa_system.get_messages")
    def test_vector_store_raises_error_on_no_documents(self, mock_get_messages, mock_exists):
        mock_get_messages.return_value = []

        with self.assertRaisesRegex(
            RuntimeError, "Failed to build vector store: No documents were found"
        ):
            rag_qa_system.VectorStore()


if __name__ == "__main__":
    unittest.main()


