import json
import os
import random
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
EN_QUESTIONS_FILE = "../../data/processed/questions/questions_EN.jsonl"
MR_QUESTIONS_FILE = "../../data/processed/questions/questions_MR.jsonl"
CHUNKS_FILE = "../../data/processed/chunks_final.jsonl"
OUTPUT_DIR = "../../data/experiment_splits/"
TRAIN_RATIO = 0.80
SEED = 42

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Step 1: Loading generated data and raw chunks...")
    en_questions = load_jsonl(EN_QUESTIONS_FILE)
    mr_questions = load_jsonl(MR_QUESTIONS_FILE)
    all_chunks = load_jsonl(CHUNKS_FILE)

    # Map article_ids to all its child chunk_ids for soft article-level grading
    article_to_chunks_map = defaultdict(list)
    chunk_text_map = {}
    
    for c in all_chunks:
        cid = c["chunk_id"]
        chunk_text_map[cid] = c["text"]
        aid = c.get("metadata", {}).get("article_id")
        if aid:
            article_to_chunks_map[aid].append(cid)

    print("Step 2: Aligning datasets using explicit Question IDs...")
    # Map Marathi questions by chunk_id to align with English generated pairs
    mr_map = {item["chunk_id"]: item for item in mr_questions}
    
    aligned_pairs = []
    for en_item in en_questions:
        cid = en_item["chunk_id"]
        if cid in mr_map and cid in chunk_text_map:
            meta = en_item.get("source", {})
            aid = meta.get("article_id")
            
            aligned_pairs.append({
                "question_id_en": en_item["question_id"],   # Uses generated ID
                "question_id_mr": mr_map[cid]["question_id"], # Uses generated ID
                "chunk_id": cid,
                "article_id": aid,
                "chunk_text": chunk_text_map[cid],
                "en_question": en_item["question"],
                "mr_question": mr_map[cid]["marathi_question"]
            })

    total_pairs = len(aligned_pairs)
    print(f"✓ Found {total_pairs} fully aligned question-chunk sets.")

    # Partition dataset into 80% Train / 20% Evaluation pools
    random.shuffle(aligned_pairs)
    split_idx = int(total_pairs * TRAIN_RATIO)
    train_data = aligned_pairs[:split_idx]
    eval_data = aligned_pairs[split_idx:]

    print(f" -> Allocation: {len(train_data)} training items | {len(eval_data)} evaluation items.")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: EXPORT TRAINING SET (For Fine-Tuning BGE-M3)
    # ─────────────────────────────────────────────────────────────
    print("Step 3: Exporting training files...")
    with open(os.path.join(OUTPUT_DIR, "train_pairs_EN.jsonl"), "w", encoding="utf-8") as f_en, \
         open(os.path.join(OUTPUT_DIR, "train_pairs_MR.jsonl"), "w", encoding="utf-8") as f_mr:
        for item in train_data:
            # Keeps files lightweight for MultipleNegativesRankingLoss processing
            f_en.write(json.dumps({"query": item["en_question"], "positive": item["chunk_text"]}) + "\n")
            f_mr.write(json.dumps({"query": item["mr_question"], "positive": item["chunk_text"]}) + "\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: GENERATE MULTI-GRADED QRELS USING UNIQUE GENERATED IDs
    # ─────────────────────────────────────────────────────────────
    print("Step 4: Compiling article-aware qrels using unique generated IDs...")
    eval_queries_en = {}
    eval_queries_mr = {}
    qrels_en = []
    qrels_mr = []
    required_eval_chunks = set()

    for item in eval_data:
        q_id_en = item["question_id_en"]
        q_id_mr = item["question_id_mr"]
        gold_chunk_id = item["chunk_id"]
        gold_article_id = item["article_id"]
        
        required_eval_chunks.add(gold_chunk_id)
        eval_queries_en[q_id_en] = item["en_question"]
        eval_queries_mr[q_id_mr] = item["mr_question"]

        # Relevance Grade 2: Exact matching original text chunk
        qrels_en.append(f"{q_id_en}\t0\t{gold_chunk_id}\t2")
        qrels_mr.append(f"{q_id_mr}\t0\t{gold_chunk_id}\t2")

        # Relevance Grade 1: Adjacent sister components within the same master document
        if gold_article_id and gold_article_id in article_to_chunks_map:
            for sister_chunk_id in article_to_chunks_map[gold_article_id]:
                if sister_chunk_id != gold_chunk_id:
                    qrels_en.append(f"{q_id_en}\t0\t{sister_chunk_id}\t1")
                    qrels_mr.append(f"{q_id_mr}\t0\t{sister_chunk_id}\t1")
                    required_eval_chunks.add(sister_chunk_id)

    # Save mapping files
    with open(os.path.join(OUTPUT_DIR, "eval_queries_EN.json"), "w", encoding="utf-8") as f:
        json.dump(eval_queries_en, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "eval_queries_MR.json"), "w", encoding="utf-8") as f:
        json.dump(eval_queries_mr, f, ensure_ascii=False, indent=2)

    # Standard TREC TSV outputs
    with open(os.path.join(OUTPUT_DIR, "qrels_EN.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(qrels_en) + "\n")
    with open(os.path.join(OUTPUT_DIR, "qrels_MR.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(qrels_mr) + "\n")

    # Isolated evaluation sub-corpus to avoid overloading database indices during testing
    eval_corpus = {cid: chunk_text_map[cid] for cid in required_eval_chunks if cid in chunk_text_map}
    with open(os.path.join(OUTPUT_DIR, "eval_corpus.json"), "w", encoding="utf-8") as f:
        json.dump(eval_corpus, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Pipeline data preparation complete! Organized files stored in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
