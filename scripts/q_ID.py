#!/usr/bin/env python3
"""
Add unique question_id to existing questions_EN.jsonl and questions_MR.jsonl.
No model loading — just hash-based ID generation from existing fields.
"""

import json

if __name__ == "__main__":
    # English questions
    chunk_file = "../data/processed/chunks_final.jsonl"
    query_file = "../data/processed/questions/questions_EN_with_ids.jsonl"
    
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]

    with open(query_file, 'r', encoding='utf-8') as f:
        query = [json.loads(line) for line in f]

    if len(chunks) == len(query):
        print("Don't worry Atul")
    else:
        print("Tuhan Deo")