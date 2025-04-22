# rag-adk-pinecone

RAG (Retrieval-Augmented Generation) implementation using Google ADK and Pinecone vector database.

## Overview

This repository demonstrates how to build a RAG system with Google's Agent Development Kit (ADK) and Pinecone for efficient vector search. The implementation provides a foundation for developing context-aware AI applications that can reference external knowledge sources.

Key features include:
- Integration of Google ADK with Pinecone vector database
- Document indexing and retrieval functionality
- Example agents that utilize RAG capabilities
- PDF document uploader for data preprocessing

## Project Structure

```
rag-adk-pinecone/
├── rag/
│   ├── __init__.py
│   ├── agent.py            # Root agent definition
│   └── tools/
│       └── tools.py        # PineconeIndexRetrieval tool implementation
├── .env.example            # Template for environment variables
├── .gitignore
├── LICENSE
├── pdf_uploader.py         # Utility for uploading PDFs to the vector database
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.9+
- Pinecone account with API key
- OpenAI API Key
- Google ADK setup with Google AI API Key
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-adk-pinecone.git
   cd rag-adk-pinecone
   ```

2. Copy the environment template and fill in your credentials:
   ```
   cp .env.example .env
   ```
   
   Edit the `.env` file to include:
   - Your API keys
   - Pinecone environment
   - Pinecone index name
   - Other required configuration

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   ```
   
   On Linux/macOS:
   ```
   source venv/bin/activate
   ```
   
   On Windows:
   ```
   .\venv\Scripts\Activate.bat
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Uploading Documents

Before running the agent, you may want to upload documents to your Pinecone index:

```
python pdf_uploader.py path/to/your/document.pdf namespace_name
```

This utility will:
1. Extract text from the PDF
2. Split the text into manageable chunks
3. Generate embeddings for each chunk
4. Upload the embeddings to your Pinecone index

### Running the Application

Start the ADK web interface:

```
adk web
```

This will launch a local web server where you can interact with your RAG agent.

## How It Works

1. **Document Indexing**: Documents are processed, split into chunks, and stored in Pinecone's vector database.

2. **Query Processing**: When a user submits a query:
   - The `PineconeIndexRetrieval` tool converts the query to an embedding
   - It searches the Pinecone index for relevant document chunks
   - The most relevant chunks are retrieved based on semantic similarity

3. **Response Generation**: The root agent in `rag/agent.py` uses the retrieved context along with the original query to generate an informed response.

## Customization

### Modifying the RAG Agent

The main agent logic is defined in `rag/agent.py`. You can modify this file to:
- Adjust the prompt template
- Change the retrieval strategy
- Implement additional business logic

### Extending Retrieval Tools

The `PineconeIndexRetrieval` tool in `rag/tools/tools.py` can be extended to:
- Support different embedding models
- Implement advanced filtering
- Add metadata-based search capabilities

## Troubleshooting

### Common Issues

- **Authentication Errors**: Ensure your API keys in the `.env` file are correct
- **Index Not Found**: Verify that your Pinecone index name matches the one in your `.env` file
- **Embedding Errors**: Check that you're using a compatible embedding model

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Agent Development Kit (ADK)
- Pinecone Vector Database
- LangChain