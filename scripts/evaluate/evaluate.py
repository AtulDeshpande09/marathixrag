#!/usr/bin/env python3
"""
Evaluate RAG retrieval across multiple k cutoffs (1, 2, 3, 5).
Outputs NDCG, Recall, and MRR.
"""

import json
import pytrec_eval

# Define file paths
QRELS_FILE = ".../data/processed/qrels.train.tsv"
RUN_FILE = ".../data/processed/results.run"

def run_evaluation():
    # 1. Load ground truth (qrels)
    print("Loading qrels...")
    with open(QRELS_FILE, 'r', encoding='utf-8') as f:
        qrels = pytrec_eval.parse_qrels(f)

    # 2. Load retriever results (run)
    print("Loading retriever run...")
    with open(RUN_FILE, 'r', encoding='utf-8') as f:
        run = pytrec_eval.parse_run(f)

    # 3. Define the exact metrics and k-cutoffs you want
    metrics = {
        # NDCG handles your multi-level scores (2 = golden, 1 = hard negative)
        'ndcg_cut_1', 'ndcg_cut_2', 'ndcg_cut_3', 'ndcg_cut_5',
        
        # Recall shows if the relevant chunk is anywhere in the top K
        'recall_1', 'recall_2', 'recall_3', 'recall_5',
        
        # Mean Reciprocal Rank (evaluates overall position of the first right answer)
        'recip_rank'
    }

    # 4. Initialize evaluator and run analysis
    print("Computing metrics...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)

    # 5. Aggregate results across all queries
    aggregated_results = {}
    sample_query = list(results.keys())[0]
    metric_names = sorted(list(results[sample_query].keys()))

    for metric in metric_names:
        avg_score = pytrec_eval.compute_aggregated_measure(metric, results.values())
        aggregated_results[metric] = avg_score

    # 6. Print beautifully structured results
    print("\n" + "="*40)
    print("      RAG RETRIEVAL EXPERIMENT RESULTS      ")
    print("="*40)
    
    print("\n--- Recall (Did we find a relevant chunk?) ---")
    print(f"Recall@1: {aggregated_results['recall_1']:.4f}")
    print(f"Recall@2: {aggregated_results['recall_2']:.4f}")
    print(f"Recall@3: {aggregated_results['recall_3']:.4f}")
    print(f"Recall@5: {aggregated_results['recall_5']:.4f}")

    print("\n--- NDCG (Did we rank golden chunks higher than hard negatives?) ---")
    print(f"NDCG@1:   {aggregated_results['ndcg_cut_1']:.4f}")
    print(f"NDCG@2:   {aggregated_results['ndcg_cut_2']:.4f}")
    print(f"NDCG@3:   {aggregated_results['ndcg_cut_3']:.4f}")
    print(f"NDCG@5:   {aggregated_results['ndcg_cut_5']:.4f}")

    print("\n--- Overall Ranking Efficiency ---")
    print(f"MRR:      {aggregated_results['recip_rank']:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
