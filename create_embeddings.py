import json
from sentence_transformers import SentenceTransformer
import chromadb
import pathlib

texts = []
metadatas = []

input_file = "data/processed/chunks.jsonl"
db = "chroma"

Path("chroma").mkdir(parents=True, exist_ok=True)

with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        texts.append(data['text'])
        metadatas.append(data['metadata'])

# Batch Embedding (Fastest way)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)

# Bulk Add to ChromaDB
client = chromadb.PersistentClient(path=db)
collection = client.get_or_create_collection(name="my_chunks")

collection.add(
    ids=[f"id_{i}" for i in range(len(texts))],
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas
)

print(f"Vector Database Saved at {db}")
