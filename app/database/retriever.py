from app.database.vector_store import collection, generate_embeddings

def retrieve_documents(query: str, top_k: int = 4):

    # Convert user question into an embedding
    query_embedding = generate_embeddings([query])[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_documents = []

    for i in range(len(results["documents"][0])):

        retrieved_documents.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved_documents


if __name__ == "__main__":

    question = "How do I define query parameters in FastAPI?"

    results = retrieve_documents(question)

    print("\nQuestion:", question)

    for i, result in enumerate(results):

        print(f"\n--- Result {i + 1} ---")
        print("Source:", result["metadata"]["source"])
        print("Distance:", result["distance"])
        print("Content:", result["content"][:300])
