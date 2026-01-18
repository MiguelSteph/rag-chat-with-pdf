import os
from typing import IO
from chunk_pdf import chunk_pdf
import tiktoken
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import QueryResult
from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

MAX_NUMBER_OF_INPUTS_PER_REQUEST = 2048
MAX_TOKENS_PER_REQUEST = 300_000


class DocumentDb:
  def __init__(self):
    self.db_path = 'chroma_db'
    self.db_client = chromadb.PersistentClient(path=self.db_path)
    self.openai_client = OpenAI(api_key=os.environ['OPEN_AI_API_KEY'])
    self.collection = self.db_client.get_or_create_collection(
          name="pdf_chunk_collection",
          embedding_function=OpenAIEmbeddingFunction(
                api_key=os.environ['OPEN_AI_API_KEY'],
                model_name=os.environ['EMBEDDING_MODEL_NAME']
              )
        )
    self.encoding = tiktoken.encoding_for_model(os.environ['EMBEDDING_MODEL_NAME'])

  
  def query(self, query: str) -> QueryResult:
    return self.collection.query(
        query_texts=[query],
        n_results=5
    )


  def handle_pdf(self, file: IO[bytes], file_name: str) -> None:
    documents = chunk_pdf(file, file_name)
    self._add_documents_to_collection(documents)


  def _add_documents_to_collection(self, docs: list[Document]) -> None:
    ids = [doc.id for doc in docs]
    embeddings = self._get_embeddings(docs)
    documents_content = [doc.page_content for doc in docs]
    documents_metadata = [doc.metadata for doc in docs]
    self.collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents_content,
        metadatas=documents_metadata,
    )


  def _get_embeddings(self, docs: list[Document]) -> list[list[float]]:
    embeddings = []
    input_arr = []
    tokens_count = 0

    def fetch_embeddings() -> None:
      embedding_response = self.openai_client.embeddings.create(
          input=input_arr,
          model=os.environ['EMBEDDING_MODEL_NAME']
      )
      request_embeddings = [data.embedding for data in embedding_response.data]
      embeddings.extend(request_embeddings)

    for doc in docs:
      current_doc_tokens_count = len(self.encoding.encode(doc.page_content))
      if (tokens_count == 0 or (current_doc_tokens_count + tokens_count) < MAX_TOKENS_PER_REQUEST) and len(input_arr) < MAX_NUMBER_OF_INPUTS_PER_REQUEST:
        input_arr.append(doc.page_content)
        tokens_count += current_doc_tokens_count
      else:
        fetch_embeddings()
        input_arr = []
        tokens_count = 0
    if len(input_arr) > 0:
      fetch_embeddings()

    return embeddings



