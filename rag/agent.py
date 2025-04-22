from google.adk.agents import Agent
from openai import OpenAI
from pinecone import Pinecone

from .tools.tools import PineconeIndexRetrieval


pinecone_tool = PineconeIndexRetrieval(
    name="pinecone_retrieval_tool",
    description='This tool retrieves data from the pinecone vector database, using "query" text input.',
    index_name="adk-rag-index",
    namespace="2024_covid19",
    pinecone=Pinecone(),
    openai_client=OpenAI(),
    top_k=5,
)

root_agent = Agent(
    name="root_agent",
    model="gemini-2.0-flash",
    description="This is root agent that retrieves data from the pinecone vector database.",
    instruction=(
        "You are a helpful assistant that retrieves data from the pinecone vector database."
        "Use the tool to retrieve the data and provide the answer to the user."
        "Please answer the question as accurately as possible."
    ),
    tools=[pinecone_tool],
)
