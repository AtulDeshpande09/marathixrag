import json
import chromadb
from FlagEmbedding import BGEM3FlagModel

# Initialize BGE-M3
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Initialize clean Chroma instance
client = chromadb.PersistentClient(path="../../chroma_clean")
collection = client.get_or_create_collection(name="my_chunks_clean")

CHUNKS_FILE = "../../data/processed/chunks_final.jsonl"

print("Indexing collection with clean structural keys...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    batch_texts = []
    batch_ids = []
    batch_metadatas = []
    
    for line in f:
        if not line.strip():
            continue
        chunk = json.loads(line.strip())
        
        batch_texts.append(chunk["text"])
        batch_ids.append(chunk["chunk_id"]) # Standardizes strict ID matching
        batch_metadatas.append({
            "article_id": chunk.get("metadata", {}).get("article_id", ""),
            "source_file": chunk.get("metadata", {}).get("source_file", "")
        })
        
        # Batch upload to ChromaDB in increments of 128 for stability
        if len(batch_ids) == 128:
            outputs = model.encode(batch_texts, batch_size=32, max_length=512)
            vectors = [vec.tolist() for vec in outputs['dense_vecs']]
            
            collection.add(
                ids=batch_ids,
                embeddings=vectors,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            batch_texts, batch_ids, batch_metadatas = [], [], []

    # Clear out any remaining items in the buffer
    if batch_ids:
        outputs = model.encode(batch_texts, batch_size=32, max_length=512)
        vectors = [vec.tolist() for vec in outputs['dense_vecs']]
        collection.add(ids=batch_ids, embeddings=vectors, documents=batch_texts, metadatas=batch_metadatas)

print("✅ ChromaDB rebuilt with exact chunk_id indexing!")
