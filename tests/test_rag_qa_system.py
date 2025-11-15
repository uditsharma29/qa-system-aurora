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


class VectorStoreTests(unittest.TestCase):
    def test_search_returns_relevant_documents(self):
        fake_messages = [
            {"user_name": "Layla", "message": "Planning a London trip next month"},
            {"user_name": "Vikram", "message": "Needs another car detail"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "vector_store.pkl")
            with mock.patch("app.rag_qa_system.CACHE_PATH", cache_path):
                with mock.patch(
                    "app.rag_qa_system.SentenceTransformer", return_value=FakeEmbedder()
                ):
                    with mock.patch(
                        "app.rag_qa_system.get_messages", return_value=fake_messages
                    ):
                        store = rag_qa_system.VectorStore()
                        results = store.search(
                            "When is Layla planning her trip to London?", k=2
                        )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].startswith("Layla"))


if __name__ == "__main__":
    unittest.main()


