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

print("")
print("Adding embeddings to VectorDB...")

collection.add(
    ids=[f"id_{i}" for i in range(len(texts))],
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas
)

print(f"Vector Database Saved at {db}")
