from __future__ import annotations

from typing import Any, Callable, override

from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
from google.adk.tools.tool_context import ToolContext
from openai import OpenAI
from pinecone import Pinecone


class PineconeIndexRetrieval(BaseRetrievalTool):
    """
    A tool for retrieving documents from a Pinecone index.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        pinecone: Pinecone,
        index_name: str,
        namespace: str,
        embedder: Callable[[str], list[float]],
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

    @override
    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        query = args.get("query")

        if not query:
            raise ValueError("Query is required")

        vector = self.embedder(query)

        results = self.index.query(
            vector=vector,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        return [result["metadata"][self.key_text] for result in results["matches"]]
