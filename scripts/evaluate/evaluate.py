import os
import json
import numpy as np
import pytrec_eval
import chromadb
from FlagEmbedding import BGEM3FlagModel
from logger import ExperimentLogger

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "../../models/bge-m3-ft-marathi"  
EVAL_DIR = "../../data/experiment_splits/"
CHROMA_DB_PATH = "../../chroma_clean"
COLLECTION_NAME = "my_chunks_clean"

K_LIST = [1, 3, 5, 10]
MAX_K = max(K_LIST)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_qrels_tsv(file_path):
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, _, doc_id, rel = line.strip().split("\t")
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = int(rel)
    return qrels

def evaluate_scenario(model, queries_dict, qrels, collection):
    run_results = {}
    
    q_ids = list(queries_dict.keys())
    q_texts = list(queries_dict.values())
    
    print(f" -> Encoding {len(q_texts)} queries...")
    embeddings_output = model.encode(q_texts, batch_size=32, max_length=512)
    query_vectors_list = [vec.tolist() for vec in embeddings_output['dense_vecs']]
    
    print(f" -> Index scan for Top-{MAX_K} matches...")
    results = collection.query(query_embeddings=query_vectors_list, n_results=MAX_K)
    
    # Standard clean IR parsing: IDs map 1:1 natively!
    for idx, qid in enumerate(q_ids):
        run_results[qid] = {}
        retrieved_ids = results["ids"][idx]
        distances = results["distances"][idx] if "distances" in results else [1.0] * len(retrieved_ids)
        
        for doc_id, dist in zip(retrieved_ids, distances):
            similarity_score = 1.0 / (1.0 + dist)
            run_results[qid][doc_id] = float(similarity_score)
            
    # Compute standard academic metrics
    metric_measures = {'map', 'recip_rank'}
    for k in K_LIST:
        metric_measures.add(f'success_{k}')
        metric_measures.add(f'ndcg_cut_{k}')
        
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metric_measures)
    scores = evaluator.evaluate(run_results)
    
    metrics = {}
    for k in K_LIST:
        metrics[f"Top-{k} Accuracy"] = np.mean([q[f"success_{k}"] for q in scores.values()])
        metrics[f"NDCG@{k}"] = np.mean([q[f"ndcg_cut_{k}"] for q in scores.values()])
    metrics["MRR"] = np.mean([q["recip_rank"] for q in scores.values()])
    metrics["MAP"] = np.mean([q["map"] for q in scores.values()])
    
    return metrics

def main():
    logger = ExperimentLogger(exp_name="BGE-M3_Fine_Tuned")
    logger.log(f"Running Clean Multi-K Evaluation for: {MODEL_PATH}")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    queries_en = load_json(os.path.join(EVAL_DIR, "eval_queries_EN.json"))
    queries_mr = load_json(os.path.join(EVAL_DIR, "eval_queries_MR.json"))
    qrels_en = load_qrels_tsv(os.path.join(EVAL_DIR, "qrels_EN.tsv"))
    qrels_mr = load_qrels_tsv(os.path.join(EVAL_DIR, "qrels_MR.tsv"))

    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)

    # SCENARIO 1: EN -> EN
    logger.section("SCENARIO 1: BASE-ENGLISH (EN -> EN)")
    metrics_en = evaluate_scenario(model, queries_en, qrels_en, collection)
    for m_name, val in metrics_en.items(): logger.log(f"{m_name}: {val:.4f}")

    # SCENARIO 2: MR -> EN
    logger.section("SCENARIO 2: BASE-MARATHI (MR -> EN)")
    metrics_mr = evaluate_scenario(model, queries_mr, qrels_mr, collection)
    for m_name, val in metrics_mr.items(): logger.log(f"{m_name}: {val:.4f}")

    # Final Matrix Print
    logger.section("FINAL RESULTS MATRIX")
    logger.log(f"{'Evaluation Metric':<25} | {'Base-English':<15} | {'Base-Marathi':<15}")
    logger.log("-" * 65)
    for m in metrics_en.keys():
        logger.log(f"{m:<25} | {metrics_en[m]:<15.4f} | {metrics_mr[m]:<15.4f}")

if __name__ == "__main__":
    main()
