import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
INPUT_FILE = ".../data/processed/chunks_final.jsonl"
OUTPUT_FILE = ".../data/processed/questions/questions_EN.jsonl"
MIN_CHUNK_LEN = 30
MAX_CHUNK_LEN = 2500
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
BATCH_SIZE = 1000  

SYSTEM_PROMPT = """You will be given a short text chunk. Generate EXACTLY ONE high-quality question that:
- Is standalone and understandable without outside context
- Has a single, clear, factual answer found directly in the chunk
- Starts with a question word: What, How, When, Where, Why, or Who

Output ONLY a valid JSON object matching the requested schema."""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"}
    },
    "required": ["question", "answer"]
}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"✗ Error: Input file not found: {INPUT_FILE}")
        return

    print("Step 1: Loading and filtering chunks...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    valid_chunks = []
    prompts = []

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunk_text = chunk.get("text", "").strip()
                
                if len(chunk_text) < MIN_CHUNK_LEN or len(chunk_text) > MAX_CHUNK_LEN:
                    continue
                
                chunk_id = chunk.get("chunk_id", "").strip()
                metadata = chunk.get("metadata", {})

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Text chunk:\n{chunk_text}"}
                ]
                
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                prompts.append(input_text)
                valid_chunks.append({"chunk_id": chunk_id, "metadata": metadata})
            except Exception as e:
                print(f"✗ Parsing error: {e}")

    total_chunks = len(valid_chunks)
    print(f"✓ Loaded {total_chunks} valid chunks.")
    
    print("Step 2: Initializing vLLM engine...")
    llm = LLM(model=MODEL_ID, quantization="awq", max_model_len=3000, gpu_memory_utilization=0.9)
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, 
        max_tokens=MAX_NEW_TOKENS,
        guided_decoding={"guided_json": json.dumps(JSON_SCHEMA)}
    )

    print("Step 3: Running chunked batch generation & writing results...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    generated_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile: # Uses "w" to ensure a fresh, clean run
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_metadata = valid_chunks[i:i + BATCH_SIZE]
            
            print(f" -> Processing batch {i // BATCH_SIZE + 1} (Chunks {i} to {min(i + BATCH_SIZE, total_chunks)})...")
            outputs = llm.generate(batch_prompts, sampling_params, display_progress=False)

            for idx, output in enumerate(outputs):
                response_text = output.outputs[0].text
                
                try:
                    qa_pair = json.loads(response_text.strip())
                    meta = batch_metadata[idx]["metadata"]
                    
                    # Incremental Question ID created here dynamically
                    q_id = f"Q_EN_{generated_count + 1:05d}"
                    
                    output_record = {
                        "question_id": q_id,  # Added Key
                        "chunk_id": batch_metadata[idx]["chunk_id"],
                        "question": qa_pair["question"].strip(),
                        "answer": qa_pair["answer"].strip(),
                        "source": {
                            "article_id": meta.get("article_id"),
                            "title": meta.get("title"),
                            "language": meta.get("language"),
                            "url": meta.get("url"),
                            "source_file": meta.get("source_file")
                        }
                    }
                    outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    generated_count += 1
                except Exception as parse_err:
                    print(f" ! Failed to parse item: {parse_err}")

    print(f"\n✅ Complete! Generated {generated_count} Q&A pairs with explicit IDs.")

if __name__ == "__main__":
    main()
