from __future__ import annotations

from typing import Any, override

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
        openai_client: OpenAI,
        top_k: int = 10,
    ):
        super().__init__(name=name, description=description)
        self.pinecone = pinecone
        self.index = self.pinecone.Index(index_name)
        self.index_name = index_name
        self.namespace = namespace
        self.openai = openai_client
        self.top_k = top_k

    def _get_embedding(self, text: str) -> list[float]:
        embedding = self.openai.embeddings.create(
            model="text-embedding-3-large", input=text
        )
        return embedding.data[0].embedding

    @override
    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        query = args.get("query")

        if not query:
            raise ValueError("Query is required")

        vector = self._get_embedding(query)

        results = self.index.query(
            vector=vector,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        return [result["metadata"]["text"] for result in results["matches"]]
