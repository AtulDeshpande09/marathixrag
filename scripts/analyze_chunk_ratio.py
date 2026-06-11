#!/usr/bin/env python3
"""
Chunk Ratio Analysis for Marathi-XRetrieval
Parses TREC-style TSV qrels and computes retrieval difficulty statistics.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_qrels_tsv(qrels_path, relevance_threshold=1):
    """
    Load TREC-style TSV qrels file.
    
    Format: qid <tab> iteration <tab> chunk_id <tab> relevance_score
    
    Args:
        qrels_path: Path to TSV qrels file
        relevance_threshold: Minimum score to count as "relevant" (default: 1)
    
    Returns:
        dict: {query_id: set(gold_chunk_ids)}
    """
    qrels = defaultdict(set)
    
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                continue
                
            qid, iteration, chunk_id, relevance = parts
            relevance = int(relevance)
            
            # Only count chunks with relevance >= threshold as "gold"
            if relevance >= relevance_threshold:
                qrels[qid].add(chunk_id)
    
    return dict(qrels)


def load_chunks_jsonl(chunks_path):
    """Load chunks JSONL: {chunk_id: {text, article_id, metadata}}"""
    chunks = {}
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            chunks[item['chunk_id']] = item
    return chunks


def compute_chunk_statistics(qrels, chunks):
    """
    Compute chunk-level statistics for retrieval difficulty analysis.
    
    Args:
        qrels: dict {query_id: set(gold_chunk_ids)}
        chunks: dict {chunk_id: {article_id, ...}}
    
    Returns:
        dict with statistics + DataFrames for visualization
    """
    # Group chunks by article
    article_chunks = defaultdict(set)
    for chunk_id, meta in chunks.items():
        article_chunks[meta['article_id']].add(chunk_id)
    
    # Compute per-article chunk distribution
    chunks_per_article = [len(c) for c in article_chunks.values()]
    
    # Compute gold chunks per query
    gold_per_query = [len(gold_set) for gold_set in qrels.values()]
    
    # Map queries to source articles (for stratification)
    query_article_map = {}
    for qid, gold_chunks in qrels.items():
        for cid in gold_chunks:
            if cid in chunks:
                query_article_map[qid] = chunks[cid]['article_id']
                break
    
    # Core statistics
    n_articles = len(article_chunks)
    n_chunks_total = len(chunks)
    n_queries = len(qrels)
    n_gold_total = sum(len(g) for g in qrels.values())
    
    stats = {
        'n_articles': n_articles,
        'n_chunks_total': n_chunks_total,
        'n_queries': n_queries,
        'n_gold_chunks_total': n_gold_total,
        'avg_chunks_per_article': round(np.mean(chunks_per_article), 2),
        'std_chunks_per_article': round(np.std(chunks_per_article), 2),
        'min_chunks_per_article': int(np.min(chunks_per_article)),
        'max_chunks_per_article': int(np.max(chunks_per_article)),
        'avg_gold_per_query': round(np.mean(gold_per_query), 2),
        'noise_ratio': f"1:{int((n_chunks_total - n_gold_total) / max(n_gold_total, 1))}",
        'retrieval_difficulty_score': round(
            np.mean(chunks_per_article) / max(np.mean(gold_per_query), 0.01), 2
        )
    }
    
    # DataFrames for visualization
    df_articles = pd.DataFrame({
        'article_id': list(article_chunks.keys()),
        'chunk_count': chunks_per_article
    })
    
    df_queries = pd.DataFrame({
        'query_id': list(qrels.keys()),
        'gold_count': gold_per_query,
        'source_article': [query_article_map.get(qid, 'unknown') for qid in qrels.keys()]
    })
    
    return stats, df_articles, df_queries


def visualize_chunk_distribution(df_articles, output_dir='figures'):
    """Generate visualizations"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Figure 1: Histogram of chunks per article
    plt.figure(figsize=(8, 5))
    sns.histplot(df_articles['chunk_count'], bins=20, kde=True, color='steelblue')
    mean_val = df_articles['chunk_count'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.1f}")
    plt.xlabel('Chunks per Article')
    plt.ylabel('Frequency')
    plt.title('Distribution of Chunk Granularity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/chunk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {output_dir}/chunk_distribution.png")


def print_report(stats):
    """Print formatted console report"""
    print("\n" + "="*60)
    print("📊 CHUNK RATIO ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📦 Corpus Overview:")
    print(f"   • Articles: {stats['n_articles']:,}")
    print(f"   • Total Chunks: {stats['n_chunks_total']:,}")
    print(f"   • Total Gold Chunks: {stats['n_gold_chunks_total']:,}")
    print(f"   • Queries: {stats['n_queries']:,}")
    
    print(f"\n🔢 Chunk Granularity:")
    print(f"   • Avg chunks/article: {stats['avg_chunks_per_article']} ± {stats['std_chunks_per_article']}")
    print(f"   • Range: [{stats['min_chunks_per_article']}, {stats['max_chunks_per_article']}]")
    
    print(f"\n🎯 Retrieval Signal:")
    print(f"   • Avg gold chunks/query: {stats['avg_gold_per_query']}")
    print(f"   • Noise Ratio: {stats['noise_ratio']} (1 signal : N distractors)")
    
    print(f"\n📈 Difficulty Metric:")
    print(f"   • Retrieval Difficulty Score: {stats['retrieval_difficulty_score']}×")
    print(f"     (Score = avg_chunks_per_article / avg_gold_per_query)")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    try:
        ratio_num = int(stats['noise_ratio'].split(':')[1])
        if ratio_num <= 3:
            print(f"   → Low noise. High Top-K may overestimate real-world performance.")
        elif ratio_num <= 10:
            print(f"   → Moderate noise. Realistic for wiki-style RAG; strong results are meaningful.")
        else:
            print(f"   → High noise. Good performance indicates robust retrieval capability.")
    except:
        print(f"   → Noise ratio indicates a challenging retrieval environment.")
    
    print("\n" + "="*60)


def export_summary(stats, output_path='../results/chunk_analysis.json'):
    """Export for reproducibility"""
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Summary exported to {output_path}")


if __name__ == "__main__":
    # ===== CONFIGURE PATHS =====
    QRELS_PATH = "../data/processed/qrels/qrels_MR.train.tsv"      # TSV format
    CHUNKS_PATH = "../data/processed/chunks.jsonl"         # JSONL format
    
    # Relevance threshold: chunks with score >= this are "gold"
    RELEVANCE_THRESHOLD = 1
    
    print("🔍 Running chunk ratio analysis (TSV qrels format)...")
    
    # Load data
    qrels = load_qrels_tsv(QRELS_PATH, relevance_threshold=RELEVANCE_THRESHOLD)
    chunks = load_chunks_jsonl(CHUNKS_PATH)
    
    print(f"   • Loaded {len(qrels)} queries from qrels")
    print(f"   • Loaded {len(chunks)} chunks from corpus")
    
    # Compute statistics
    stats, df_articles, df_queries = compute_chunk_statistics(qrels, chunks)
    
    # Generate visualizations
    visualize_chunk_distribution(df_articles)
    
    # Print report
    print_report(stats)
    
    # Export for reproducibility
    export_summary(stats)
    
    print("\n✅ Analysis complete. Use these stats to contextualize retrieval results.")