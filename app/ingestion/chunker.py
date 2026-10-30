from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.loader import load_documents
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    documents = load_documents()

    chunks = split_documents(documents)

    print("Total documents:", len(documents))
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:5]):

        print(f"\n--- Chunk {i + 1} ---")
        print("Source:", chunk.metadata["source"])
        print("Length:", len(chunk.page_content))
        print("Content:", chunk.page_content[:200])
