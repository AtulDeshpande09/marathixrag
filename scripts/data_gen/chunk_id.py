#!/usr/bin/env python3
"""
Add chunk_id to each chunk based on article_id.
Format: {article_id}_chunk_{N:05d} where N resets per article.
"""

import json
from collections import defaultdict

INPUT = "../data/processed/chunks.jsonl"
OUTPUT = "../data/processed/chunks_final.jsonl"

# Track counter per article_id
counter = defaultdict(int)

with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    for line in fin:
        chunk = json.loads(line.strip())
        article_id = chunk["metadata"]["article_id"]
        
        # Increment counter for this article only
        counter[article_id] += 1
        
        # Assign chunk_id
        chunk["chunk_id"] = f"{article_id}_chunk_{counter[article_id]:05d}"
        
        # Write with chunk_id added
        fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"✅ Done: {OUTPUT}")
print(f"📊 Processed {sum(counter.values()):,} chunks from {len(counter)} articles")