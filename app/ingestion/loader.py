
from pathlib import Path
from langchain_core.documents import Document

DOCUMENTS_DIR = Path("documents/fastapi")


def load_documents():
    documents = []

    # Find all Markdown files
    for file_path in DOCUMENTS_DIR.glob("*.md"):
        # Read file content
        content = file_path.read_text(encoding="utf-8")
        # Create a Document object
        document = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "category": "FastAPI Documentation"
            }
        )

        documents.append(document)

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"Total documents loaded: {len(docs)}")

    for doc in docs:
        print("\nSource:", doc.metadata["source"])
        print("Characters:", len(doc.page_content))
        print("Preview:", doc.page_content[:150])
