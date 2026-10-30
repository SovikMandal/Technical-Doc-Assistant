import chromadb
from app.config import CHROMA_DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embedding_model


def generate_embeddings(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()