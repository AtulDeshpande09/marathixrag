import os
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
INPUT_FILE = "../../data/processed/chunks_final.jsonl"
OUTPUT_FILE = "../../data/processed/questions/questions_EN.jsonl"
MIN_CHUNK_LEN = 30
MAX_CHUNK_LEN = 2500
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You will be given a short text chunk. Generate EXACTLY ONE high-quality question that:
- Is standalone and understandable without outside context
- Has a single, clear, factual answer found directly in the chunk
- Starts with a question word: What, How, When, Where, Why, or Who

Output ONLY a valid JSON object matching the requested schema."""

# Python dictionary passed directly to StructuredOutputsParams
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

    print("Step 1: Initializing vLLM engine...")
    # Initialize LLM first to extract its internal optimized tokenizer
    llm = LLM(model=MODEL_ID, quantization="awq", max_model_len=3000, gpu_memory_utilization=0.9)
    tokenizer = llm.get_tokenizer()

    print("Step 2: Loading and filtering chunks...")
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
    
    # Configure the updated modern structured outputs parameter syntax
    structured_params = StructuredOutputsParams(json=JSON_SCHEMA)

    sampling_params = SamplingParams(
        temperature=TEMPERATURE, 
        max_tokens=MAX_NEW_TOKENS,
        structured_outputs=structured_params
    )

    print("Step 3: Running engine-managed global batch generation...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    generated_count = 0

    # Passing the full array directly allows vLLM to utilize its continuous batching loop
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    print("Step 4: Parsing results and writing to disk...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for idx, output in enumerate(outputs):
            response_text = output.outputs[0].text
            
            try:
                qa_pair = json.loads(response_text.strip())
                meta = valid_chunks[idx]["metadata"]
                
                q_id = f"Q_EN_{generated_count + 1:05d}"
                
                output_record = {
                    "question_id": q_id,
                    "chunk_id": valid_chunks[idx]["chunk_id"],
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
                print(f" ! Failed to parse item index {idx}: {parse_err}")

    print(f"\n✅ Complete! Generated {generated_count} Q&A pairs with explicit IDs.")

if __name__ == "__main__":
    main()
