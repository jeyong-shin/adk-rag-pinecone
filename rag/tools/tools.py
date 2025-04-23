from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeAlias

from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
from google.adk.tools.tool_context import ToolContext
from pinecone import Pinecone

logger = logging.getLogger(__name__)

EmbedderFn: TypeAlias = Callable[[str], list[float]]


class PineconeIndexRetrieval(BaseRetrievalTool):
    """
    A robust tool for retrieving documents from a Pinecone index using vector similarity.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        pinecone: Pinecone,
        index_name: str,
        namespace: str,
        embedder: EmbedderFn,
        top_k: int = 10,
        key_text: str = "text",
    ):
        super().__init__(name=name, description=description)
        self.pinecone = pinecone
        self.index = self.pinecone.Index(index_name)
        self.index_name = index_name
        self.namespace = namespace
        self.embedder = embedder
        self.top_k = top_k
        self.key_text = key_text

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context: Optional[ToolContext] = None,
    ) -> list[str]:
        query = args.get("query")

        if not isinstance(query, str) or not query.strip():
            logger.warning("Invalid or missing query")
            raise ValueError("Query must be a non-empty string.")

        logger.info(f"Running Pinecone retrieval for query: {query!r}")

        try:
            vector = self.embedder(query)
        except Exception as e:
            logger.error("Failed to generate embedding vector", exc_info=True)
            raise RuntimeError("Embedder failed to generate a valid vector.") from e

        if not isinstance(vector, list) or not all(
            isinstance(x, float) for x in vector
        ):
            raise TypeError("Embedder must return a list of floats.")

        logger.debug(f"Embedding vector generated (dim={len(vector)})")

        try:
            results = self.index.query(
                vector=vector,
                top_k=self.top_k,
                namespace=self.namespace,
                include_metadata=True,
            )
        except Exception as e:
            logger.error("Pinecone query failed", exc_info=True)
            raise RuntimeError("Failed to query Pinecone index.") from e

        matches = results.get("matches", [])
        texts = [
            match["metadata"][self.key_text]
            for match in matches
            if isinstance(match.get("metadata", {}).get(self.key_text), str)
        ]

        logger.info(f"Retrieved {len(texts)} results from Pinecone.")

        return texts
