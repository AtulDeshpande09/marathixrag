from FlagEmbedding import BGEM3FlagModel
import json
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
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 
embeddings = model.encode(texts, batch_size=12, max_length=8192)['dense_vecs']


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
