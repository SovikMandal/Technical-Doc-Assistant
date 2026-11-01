import hashlib
from langchain_core.documents import Document
from app.ingestion.loader import load_documents
from app.ingestion.chunker import split_documents
from app.database.vector_store import collection, generate_embeddings


def _chunk_id(source: str, index: int) -> str:
    """Build a chunk ID that is unique per source and stable across runs."""

    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"{source_hash}_{index}"


def index_chunks(documents: list[Document]) -> dict[str, int]:
    """Chunk, embed, and store documents.

    Re-ingesting a source replaces its previous chunks, so edited or
    shortened documents do not leave stale chunks behind.

    Returns the number of chunks stored for each source.
    """

    chunks = split_documents(documents)

    if not chunks:
        print("No document chunks found.")
        return {}

    chunks_by_source: dict[str, list[Document]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.metadata["source"], []).append(chunk)

    ids, texts, metadatas = [], [], []
    for source, source_chunks in chunks_by_source.items():
        for index, chunk in enumerate(source_chunks):
            ids.append(_chunk_id(source, index))
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)

    # Embed before deleting so a model failure leaves the index unchanged
    embeddings = generate_embeddings(texts)

    for source in chunks_by_source:
        collection.delete(where={"source": source})

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Indexing completed!")
    print("Total chunks:", len(chunks))
    print("Stored chunks:", collection.count())

    return {
        source: len(source_chunks)
        for source, source_chunks in chunks_by_source.items()
    }


def index_documents() -> dict[str, int]:
    """Index the predefined local Markdown corpus."""

    return index_chunks(load_documents())


if __name__ == "__main__":
    index_documents()
