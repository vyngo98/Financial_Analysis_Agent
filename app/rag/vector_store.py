from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)


def get_embeddings():

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_client():

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )


def get_vector_store():

    embeddings = get_embeddings()
    client = get_client()

    if not client.collection_exists(QDRANT_COLLECTION):
        # Probe the embedding dimension so we don't hardcode it (and stay
        # correct if EMBEDDING_MODEL changes).
        dim = len(embeddings.embed_query("dimension probe"))

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )

        # Index the field we filter on so document isolation stays fast.
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="metadata.document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )


def build_document_filter(document_id: str):
    """Qdrant filter isolating a single document.

    QdrantVectorStore stores LangChain metadata under the "metadata" payload
    key, so the field path is "metadata.document_id".
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )
