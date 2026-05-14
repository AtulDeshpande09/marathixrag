import json
from FlagEmbedding import BGEM3FlagModel
import chromadb

# Configuration
QUESTIONS_FILE = "marathi_questions.jsonl"
K_VALUE = 5  # Benchmark Top-5 accuracy

total_queries = 0
successful_hits = 0

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 

# Initialize Chroma Native Client
client = chromadb.PersistentClient(path="chroma")
collection = client.get_collection(name="my_chunks")

with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        target_filename = data["filename"] # e.g., "en_010148f3.json"
        
        marathi_block = data.get("questions_marathi", "")
        questions = [q.strip() for q in marathi_block.split('\n') if q.strip()]
        
        for query_text in questions:
            total_queries += 1
            
            # Generate the matching dense vector format
            query_output = model.encode([query_text], batch_size=1, max_length=512)
            query_vector = query_output['dense_vecs'][0].tolist()
            
            # Query Chroma with the vector
            search_results = collection.query(
                query_embeddings=[query_vector],
                n_results=K_VALUE
            )
            
            # Parse the metadata list
            retrieved_metadatas = search_results["metadatas"][0]
            matched_source_files = [meta["source_file"] for meta in retrieved_metadatas]
            
            # Validate matching file context
            is_hit = any(target_filename in source_path for source_path in matched_source_files)
            
            if is_hit:
                successful_hits += 1

# Calculate final metrics
if total_queries > 0:
    hit_rate = (successful_hits / total_queries) * 100
    print(f"=== X-RAG Evaluation Complete ===")
    print(f"Total Marathi Queries Evaluated: {total_queries}")
    print(f"Top-{K_VALUE} Retrieval Accuracy: {hit_rate:.2f}%")


# Logging the resuls
from logger import ExperimentLogger
logger = ExperimentLogger(f"top_k_{K_VALUE}_run_1")

logger.section("Results")
logger.log(f"Total Marathi Queries Evaluated: {total_queries}")
logger.log(f"Top-{K_VALUE} Retrieval Accuracy: {hit_rate:.2f}%")
     
