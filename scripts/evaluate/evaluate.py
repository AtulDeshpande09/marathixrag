import os
import json
import numpy as np
import pytrec_eval
import chromadb
from FlagEmbedding import BGEM3FlagModel

# Import your logger class
from logger_util import ExperimentLogger

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "BAAI/bge-m3"  # Will point to local directory for FT run later
EVAL_DIR = ".../data/experiment_splits/"
CHROMA_DB_PATH = ".../chroma"
COLLECTION_NAME = "my_chunks"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_qrels_tsv(file_path):
    """Parses standard multi-graded TSV file into pytrec_eval dictionary structure."""
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, _, doc_id, rel = line.strip().split("\t")
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = int(rel)
    return qrels

def evaluate_scenario(model, queries_dict, qrels, collection, top_k=5):
    """Runs parallel verification loops using your FlagEmbedding vector strategy."""
    run_results = {}
    
    q_ids = list(queries_dict.keys())
    q_texts = list(queries_dict.values())
    
    print(f" -> Processing {len(q_texts)} sequences using FlagEmbedding...")
    
    # Utilizing your precise FlagEmbedding call layout
    # Running as a whole-array batch is significantly faster than single-line loops
    embeddings_output = model.encode(q_texts, batch_size=32, max_length=512)
    dense_vectors = embeddings_output['dense_vecs']
    
    print(" -> Scanning ChromaDB indices...")
    # Convert numpy output collection array to plain lists for Chroma compliance
    query_vectors_list = [vec.tolist() for vec in dense_vectors]
    
    results = collection.query(
        query_embeddings=query_vectors_list,
        n_results=top_k
    )
    
    # Process outputs into pytrec_eval matching keys
    for idx, qid in enumerate(q_ids):
        run_results[qid] = {}
        retrieved_ids = results["ids"][idx]
        distances = results["distances"][idx] if "distances" in results else [1.0] * len(retrieved_ids)
        
        for doc_id, dist in zip(retrieved_ids, distances):
            # Normalize distances to ascending similarity rankings
            similarity_score = 1.0 / (1.0 + dist)
            run_results[qid][doc_id] = float(similarity_score)
            
    # Calculate your strict and soft publication-ready metrics
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'ndcg_cut_5', 'map', 'recip_rank', 'success_1', 'success_3', 'success_5'}
    )
    scores = evaluator.evaluate(run_results)
    
    metrics = {
        "Top-1 Accuracy": np.mean([q["success_1"] for q in scores.values()]),
        "Top-3 Accuracy": np.mean([q["success_3"] for q in scores.values()]),
        "Top-5 Accuracy": np.mean([q["success_5"] for q in scores.values()]),
        "MRR": np.mean([q["recip_rank"] for q in scores.values()]),
        "NDCG@5": np.mean([q["ndcg_cut_5"] for q in scores.values()]),
        "MAP": np.mean([q["map"] for q in scores.values()]),
    }
    return metrics

def main():
    # Instantiate your custom logger
    logger = ExperimentLogger(exp_name="BGE-M3_Native_Baseline")
    logger.log(f"Initializing Native FlagEmbedding Verification Loop for: {MODEL_PATH}")
    
    print("Step 1: Instantiating persistent Chroma Client...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    print("Step 2: Loading query maps and truth qrels records...")
    queries_en = load_json(os.path.join(EVAL_DIR, "eval_queries_EN.json"))
    queries_mr = load_json(os.path.join(EVAL_DIR, "eval_queries_MR.json"))
    qrels_en = load_qrels_tsv(os.path.join(EVAL_DIR, "qrels_EN.tsv"))
    qrels_mr = load_qrels_tsv(os.path.join(EVAL_DIR, "qrels_MR.tsv"))

    print("Step 3: Loading original weights using native library...")
    # Matches your exact configuration flag parameters
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)

    # ─────────────────────────────────────────────────────────────
    # EXECUTE BASE-ENGLISH (EN -> EN)
    # ─────────────────────────────────────────────────────────────
    logger.section("SCENARIO 1: BASE-ENGLISH (EN -> EN)")
    print("\nRunning Base-English Evaluation...")
    metrics_en = evaluate_scenario(model, queries_en, qrels_en, collection, top_k=K_VALUE)
    for m_name, val in metrics_en.items():
        logger.log(f"{m_name}: {val:.4f}")

    # ─────────────────────────────────────────────────────────────
    # EXECUTE BASE-MARATHI (MR -> EN)
    # ─────────────────────────────────────────────────────────────
    logger.section("SCENARIO 2: BASE-MARATHI (MR -> EN)")
    print("\nRunning Base-Marathi Evaluation...")
    metrics_mr = evaluate_scenario(model, queries_mr, qrels_mr, collection, top_k=K_VALUE)
    for m_name, val in metrics_mr.items():
        logger.log(f"{m_name}: {val:.4f}")

    # Output consolidated summary matrix text block
    logger.section("FINAL BASELINE RESULTS MATRIX")
    logger.log(f"{'Evaluation Metric':<20} | {'Base-English':<15} | {'Base-Marathi':<15}")
    logger.log("-" * 60)
    for metric in metrics_en.keys():
        logger.log(f"{metric:<20} | {metrics_en[metric]:<15.4f} | {metrics_mr[metric]:<15.4f}")

    print(f"\n✅ Native evaluation pipeline completed. Metrics written securely to experimental log logs.")

if __name__ == "__main__":
    main()
