#!/usr/bin/env python3
"""
Generate TREC-style qrels using question_id as query and chunk_id as doc.
Format: query_id 0 doc_id relevance
"""

import json
from collections import defaultdict

QUESTIONS_FILE = "../data/processed/questions_MR_with_ids.jsonl"  # or _MR
CHUNKS_FILE = "../data/processed/chunks.jsonl"
OUTPUT_QRELS = "../data/processed/qrels.train.tsv"

# 1. Build article_id → [chunk_ids] map
print("Indexing chunks by article...")
article_to_chunks = defaultdict(list)
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        chunk = json.loads(line.strip())
        article_id = chunk.get("metadata", {}).get("article_id")
        # Match your chunk_id format from generate_questions.py
        chunk_id = f"{article_id}_chunk_{i:05d}"
        if article_id:
            article_to_chunks[article_id].append(chunk_id)

# 2. Generate qrels
print(f"Generating qrels from {QUESTIONS_FILE}...")
with open(QUESTIONS_FILE, "r", encoding="utf-8") as qf, \
     open(OUTPUT_QRELS, "w", encoding="utf-8") as out:
    
    for line in qf:
        record = json.loads(line.strip())
        
        query_id = record["question_id"]           # ← The question
        chunk_id = record["chunk_id"]              # ← The POSITIVE chunk (where answer lives)
        article_id = record.get("source", {}).get("article_id")
        
        # ✅ Positive: the source chunk (relevance = 2)
        out.write(f"{query_id}\t0\t{chunk_id}\t2\n")
        
        # ⚠️ Hard negatives: other chunks from same article (relevance = 1)
        if article_id and article_id in article_to_chunks:
            for other_chunk in article_to_chunks[article_id]:
                if other_chunk != chunk_id:  # Skip the positive
                    out.write(f"{query_id}\t0\t{other_chunk}\t1\n")

print(f"✅ Qrels saved to {OUTPUT_QRELS}")