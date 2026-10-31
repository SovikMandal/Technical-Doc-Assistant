from app.ingestion.loader import load_documents
from app.ingestion.chunker import split_documents
from app.database.vector_store import collection, generate_embeddings


def index_documents():
    """Index the predefined local Markdown corpus."""

    documents = load_documents()
    chunks = split_documents(documents)

    if not chunks:
        print("No document chunks found.")
        return

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    embeddings = generate_embeddings(texts)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Indexing completed!")
    print("Total chunks:", len(chunks))
    print("Stored chunks:", collection.count())


if __name__ == "__main__":
    index_documents()
