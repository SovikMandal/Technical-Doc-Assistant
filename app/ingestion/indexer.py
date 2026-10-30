import hashlib
import uuid
from html.parser import HTMLParser
from pathlib import Path
from langchain_core.documents import Document
from app.ingestion.loader import load_documents
from app.ingestion.chunker import split_documents
from app.database.vector_store import collection, generate_embeddings

class HTMLTextExtractor(HTMLParser):
    """Extract readable text from an HTML document."""

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self):
        return "\n".join(self.parts)


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


def index_uploaded_documents(uploaded_files: list[tuple[str, bytes]]):
    """Safely index uploaded Markdown, text, and HTML files."""

    documents = []

    for filename, file_bytes in uploaded_files:
        extension = Path(filename).suffix.lower()

        try:
            content = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{filename} must be a valid UTF-8 text file."
            ) from error

        if extension == ".html":
            parser = HTMLTextExtractor()
            parser.feed(content)
            content = parser.get_text()

        if not content.strip():
            raise ValueError(f"{filename} contains no readable text.")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": f"uploads/{filename}",
                    "category": "Uploaded Documentation"
                }
            )
        )

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No document chunks could be created.")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Generate embeddings before modifying existing records.
    embeddings = generate_embeddings(texts)

    uploaded_sources = list({
        document.metadata["source"] for document in documents
    })

    # Generate temporary unique IDs for the new chunks.
    batch_id = uuid.uuid4().hex

    temporary_ids = []

    for index, chunk in enumerate(chunks):
        source = chunk.metadata["source"]
        content = chunk.page_content

        unique_value = f"{source}:{index}:{content}"
        digest = hashlib.sha256(
            unique_value.encode("utf-8")
        ).hexdigest()

        temporary_ids.append(
            f"upload_{digest}_{batch_id}"
        )

    # Insert the new chunks before deleting the old ones.
    try:
        collection.add(
            ids=temporary_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    except Exception:
        # Remove any partially inserted temporary records.
        collection.delete(ids=temporary_ids)
        raise

    # New chunks are stored. Now replace the previous versions.
    try:
        for source in uploaded_sources:
            collection.delete(where={"source": source})

        # Remove temporary IDs and promote the new chunks to stable IDs.
        stable_ids = []

        for index, chunk in enumerate(chunks):
            source = chunk.metadata["source"]
            content = chunk.page_content

            unique_value = f"{source}:{index}:{content}"
            digest = hashlib.sha256(
                unique_value.encode("utf-8")
            ).hexdigest()

            stable_ids.append(f"upload_{digest}")

        collection.upsert(
            ids=stable_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        collection.delete(ids=temporary_ids)

    except Exception:
        # Preserve the temporary copies if promotion fails.
        # They contain the successfully indexed replacement documents.
        raise

    return {
        "uploaded_files": len(uploaded_files),
        "indexed_chunks": len(chunks),
        "total_chunks_in_database": collection.count()
    }


if __name__ == "__main__":
    index_documents()
