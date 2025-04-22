import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def sanitize_namespace(file_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ascii_namespace = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name)
    return ascii_namespace


def load_pdf_chunks(file_path: str) -> List[str]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return chunks


def embed_and_upsert(chunks: List[str], namespace: str):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY
    )
    texts = [doc.page_content for doc in chunks]
    metadata = [doc.metadata for doc in chunks]

    def embed_one(i: int):
        text = texts[i]
        embedding = embeddings.embed_query(text)
        return {
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": metadata[i] | {"text": text},
        }

    print("🧠 Embedding 병렬 처리 시작...")
    vectors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(embed_one, i) for i in range(len(texts))]
        for future in as_completed(futures):
            vectors.append(future.result())

    print(f"🔌 Pinecone에 연결 중...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(name=PINECONE_INDEX_NAME)

    print(f"📤 {len(vectors)}개의 벡터 업서트 중...")
    index.upsert(vectors=vectors, namespace=namespace)
    print(f"✅ 업서트 완료! namespace = '{namespace}'")


def main():
    if len(sys.argv) != 3:
        print("사용법: python pdf_uploader.py [파일경로] [업서트할 네임스페이스]")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return

    namespace = sys.argv[2]
    if not re.match(r"^[a-zA-Z0-9_-]+$", namespace):
        print(
            "❌ 네임스페이스는 알파벳, 숫자, 언더스코어(_) 및 하이픈(-)만 포함할 수 있습니다."
        )
        return

    print(f"📄 PDF 로딩: {file_path}")
    chunks = load_pdf_chunks(file_path)
    print(f"🔗 Chunk 개수: {len(chunks)}")

    if len(chunks) == 0:
        print("❌ PDF에서 추출된 chunk가 없습니다.")
        return

    embed_and_upsert(chunks, namespace)


if __name__ == "__main__":
    main()
