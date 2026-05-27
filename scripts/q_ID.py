#!/usr/bin/env python3
"""
Add unique question_id to existing questions_EN.jsonl and questions_MR.jsonl.
No model loading — just hash-based ID generation from existing fields.
"""

import json
import hashlib
import os

def make_question_id(chunk_id, question_text, answer_text="", index=0):
    """Generate a short, deterministic question_id from content."""
    raw = f"{chunk_id}:{question_text.strip()}:{answer_text.strip()}:{index}"
    hash_part = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"qid_{hash_part}"

def add_question_ids(input_file, output_file, is_marathi=False):
    """Read JSONL, add question_id, write new file."""
    print(f"Processing {input_file} → {output_file}")
    
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for i, line in enumerate(infile, 1):
            record = json.loads(line.strip())
            
            # Get fields for ID generation
            chunk_id = record.get("chunk_id", record.get("source", {}).get("article_id", "unknown"))
            
            if is_marathi:
                # MR file: has marathi_question, but we match on original English question via chunk_id
                # For now, use marathi_question + chunk_id for uniqueness
                question_text = record.get("marathi_question", "")
                answer_text = record.get("answer", "")
            else:
                # EN file: has question field
                question_text = record.get("question", "")
                answer_text = record.get("answer", "")
            
            # Generate ID
            qid = make_question_id(chunk_id, question_text, answer_text)
            
            # Add question_id to record
            record["question_id"] = qid
            
            # Write updated record
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            if i % 1000 == 0:
                print(f"  → Processed {i} records...")
    
    print(f"✅ Done! Saved to {output_file}\n")

# ─────────────────────────────────────
# RUN
# ─────────────────────────────────────
if __name__ == "__main__":
    # English questions
    add_question_ids(
        "../data/processed/questions_EN.jsonl",
        "../data/processed/questions_EN_with_ids.jsonl",
        is_marathi=False
    )
    
    # Marathi questions (if file exists)
    mr_input = "../data/processed/questions_MR.jsonl"
    if os.path.exists(mr_input):
        add_question_ids(
            mr_input,
            "../data/processed/questions_MR_with_ids.jsonl",
            is_marathi=True
        )
    else:
        print("⚠ Marathi file not found yet — skip for now")
    
    print("💡 Tip: Use questions_*_with_ids.jsonl for qrels generation")