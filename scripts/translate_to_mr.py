import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_ID = "sarvamai/sarvam-translate"
INPUT_FILE = "../data/processed/questions/questions_EN.jsonl"   # Input data
OUTPUT_FILE = "../data/processed/questions/questions_MR.jsonl"  # Output data

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
                if "question" in record and "chunk_id" in record:
                    # Configured using Sarvam's official context structure for high fidelity
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

    print(f"✓ Loaded {len(valid_records)} questions to translate.")
    
    print("Step 2: Spinning up the vLLM batch engine...")
    # Model fits comfortably in VRAM; max context length set to safe 4096 limit
    llm = LLM(model=MODEL_ID, max_model_len=4096, dtype="bfloat16", gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0.01, max_tokens=256)

    print("Step 3: Launching ultra-fast parallel translation pipeline...")
    outputs = llm.generate(prompts, sampling_params)

    print("Step 4: Structuring minimal records and writing output...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for idx, output in enumerate(outputs):
            translated_text = output.outputs.text.strip()
            orig_record = valid_records[idx]
            
            # Formats minimal structural record keeping original properties intact
            minimal_record = {
                "chunk_id": orig_record["chunk_id"],
                "marathi_question": translated_text,
                "answer": orig_record["answer"],
                "source": orig_record.get("source", {})
            }
            outfile.write(json.dumps(minimal_record, ensure_ascii=False) + "\n")

    print(f"\n✅ Pipeline Complete! Output archived at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
