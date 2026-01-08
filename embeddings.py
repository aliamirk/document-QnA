import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from process_pdf import generateChunks

chunks = generateChunks()

# Step 1: Load MiniLM model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Prepare texts from chunks
texts = [chunk["text"] for chunk in chunks]

# Step 3: Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings, dtype="float32")
print(f"Generated embeddings shape: {embeddings.shape}")

# Step 4: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

# Step 5: Save the index
faiss.write_index(index, "faiss_index.index")
print("FAISS index saved as faiss_index.index")

# Optional: Save mapping of chunk metadata
with open("chunks_metadata.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("Chunk metadata saved as chunks_metadata.pkl")

# Step 6: Example search function

def search(query, top_k=3):
    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True)

    # Ensure float32 for FAISS
    query_vec = np.array(query_vec, dtype=np.float32)

    # Search FAISS
    D, I = index.search(query_vec, top_k)

    results = []
    for idx in I[0]:
        results.append(chunks[idx])
    return results


# Example usage
if __name__ == "__main__":
    query = "What is liquidity in stock market?"
    results = search(query)
    for r in results:
        print(
            f"Page {r['page']} | Chunk {r['chunk_id']}: {r['text'][:150]}...\n")
