import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"          # High-speed 4-bit version
INPUT_FILE = "../data/processed/chunks_final.jsonl"
OUTPUT_FILE = "../data/processed/questions/questions_EN.jsonl"
MIN_CHUNK_LEN = 30
MAX_CHUNK_LEN = 2500
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You will be given a short text chunk. Generate EXACTLY ONE high-quality question that:
- Is standalone and understandable without outside context
- Has a single, clear, factual answer found directly in the chunk
- Starts with a question word: What, How, When, Where, Why, or Who

Output ONLY a valid JSON object with exactly two keys: "question" and "answer".
Do NOT include markdown, explanations, or extra text.
If no valid question can be generated, output null.

Example: {"question": "When was Don Powell born?", "answer": "10 September 1946"}"""

def parse_llm_response(response_text):
    cleaned = response_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict) and "question" in result and "answer" in result:
            return {
                "question": result["question"].strip(),
                "answer": result["answer"].strip()
            }
    except:
        pass
    return None

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

    print(f"✓ Loaded {len(valid_chunks)} valid chunks.")
    print("Step 2: Initializing vLLM engine...")
    
    # Initialize vLLM with AWQ quantization support
    llm = LLM(model=MODEL_ID, quantization="awq", max_model_len=4096, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)

    print("Step 3: Running parallel batch generation...")
    outputs = llm.generate(prompts, sampling_params)

    print("Step 4: Parsing and writing results...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    generated_count = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        for idx, output in enumerate(outputs):
            response_text = output.outputs[0].text
            qa_pair = parse_llm_response(response_text)
            
            if qa_pair:
                meta = valid_chunks[idx]["metadata"]
                output_record = {
                    "chunk_id": valid_chunks[idx]["chunk_id"],
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
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

    print(f"\n✅ Complete! Generated {generated_count} Q&A pairs.")

if __name__ == "__main__":
    main()
