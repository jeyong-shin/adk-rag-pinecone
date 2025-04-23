import os
from dotenv import load_dotenv

from google.adk.agents import Agent
from openai import OpenAI
from pinecone import Pinecone

from .tools.tools import PineconeIndexRetrieval

load_dotenv()

PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE")
if not PINECONE_NAMESPACE:
    raise ValueError("PINECONE_NAMESPACE environment variable is not set.")

pinecone_tool = PineconeIndexRetrieval(
    name="pinecone_retrieval_tool",
    description='This tool retrieves data from the pinecone vector database.',
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
    namespace=PINECONE_NAMESPACE,
    pinecone=Pinecone(),
    openai_client=OpenAI(),
    openai_embedding_model="text-embedding-3-large",
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
