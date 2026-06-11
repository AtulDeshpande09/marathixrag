from FlagEmbedding import BGEM3FlagModel
import json
import chromadb
from pathlib import Path

texts = []
metadatas = []

input_file = "data/processed/chunks.jsonl"
db = "chroma"

Path("chroma").mkdir(parents=True, exist_ok=True)

print("Collecting TEXTs and METADATAs...")

with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        texts.append(data['text'])
        metadatas.append(data['metadata'])

print("Done!!!")
print("--*--*"*5)
print("Loading Embedding Model...")
# Batch Embedding (Fastest way)
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 

print("--*--*"*5)
print("Creating Vector Embeddings...")
embeddings = model.encode(texts, batch_size=12, max_length=8192)['dense_vecs']
print("complete!!!")

# Bulk Add to ChromaDB
client = chromadb.PersistentClient(path=db)
collection = client.get_or_create_collection(name="my_chunks")

BATCH_SIZE = 5000
ids = [f"id_{i}" for i in range(len(texts))]


print("")
print("Adding embeddings to VectorDB...")


for i in range(0 , len(ids), BATCH_SIZE):
    
    batch_ids = ids[i : i + BATCH_SIZE]
    batch_embeddings = embeddings[i : i+BATCH_SIZE].tolist()
    batch_texts = texts[i : i + BATCH_SIZE]
    batch_metadatas = metadatas[i : i + BATCH_SIZE]

    collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
            )

    print(f"Uploaded batch {i//BATCH_SIZE + 1} ({len(batch_ids)} items)")


print(f"Vector Database Saved at {db}")
