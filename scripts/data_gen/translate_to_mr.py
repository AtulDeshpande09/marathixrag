import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_ID = "sarvamai/sarvam-translate"
INPUT_FILE = "../data/processed/questions/questions_EN.jsonl"   
OUTPUT_FILE = "../data/processed/questions/questions_MR.jsonl"  
BATCH_SIZE = 1000  

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"✗ Error: Input file not found: {INPUT_FILE}")
        return

    print("Step 1: Reading English entries and compiling prompt templates...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    valid_records = []
    prompts = []

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "question" in record and "question_id" in record: # Enforces checking for the new ID
                    messages = [
                        {"role": "system", "content": "Translate the text below to Marathi."},
                        {"role": "user", "content": record["question"]}
                    ]
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(input_text)
                    valid_records.append(record)
            except Exception as e:
                print(f"✗ Parsing error on input row: {e}")

    total_records = len(valid_records)
    print(f"✓ Loaded {total_records} questions to translate.")
    
    print("Step 2: Spinning up the vLLM batch engine...")
    llm = LLM(model=MODEL_ID, max_model_len=1024, dtype="bfloat16", gpu_memory_utilization=0.9, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.01, max_tokens=256)

    print("Step 3: Running chunked batch translation pipeline & writing results...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for i in range(0, total_records, BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_records = valid_records[i:i + BATCH_SIZE]
            
            print(f" -> Processing translation batch {i // BATCH_SIZE + 1} (Items {i} to {min(i + BATCH_SIZE, total_records)})...")
            outputs = llm.generate(batch_prompts, sampling_params, display_progress=False)

            for idx, output in enumerate(outputs):
                translated_text = output.outputs[0].text.strip()
                orig_record = batch_records[idx]
                
                # Replaces 'EN' suffix with 'MR' for paired matching
                mr_qid = orig_record["question_id"].replace("EN", "MR")
                
                minimal_record = {
                    "question_id": mr_qid,  # Added Key
                    "chunk_id": orig_record["chunk_id"],
                    "marathi_question": translated_text,
                    "answer": orig_record["answer"],
                    "source": orig_record.get("source", {})
                }
                outfile.write(json.dumps(minimal_record, ensure_ascii=False) + "\n")

    print(f"\n✅ Pipeline Complete! Output archived at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
