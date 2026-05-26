import os
import json
import hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "../data/processed/chunks.jsonl"      # Your chunked data
OUTPUT_FILE = "../data/processed/questions_EN.jsonl"          # One JSON object per question
MIN_CHUNK_LEN = 50                                 # Skip tiny chunks
MAX_CHUNK_LEN = 2000                               # Skip overly long chunks
MAX_NEW_TOKENS = 256                               # Output length for 1 Q&A
TEMPERATURE = 0.1                                  # Deterministic output



SYSTEM_PROMPT = """You will be given a short text chunk. Generate EXACTLY ONE high-quality question that:
- Is standalone and understandable without outside context
- Has a single, clear, factual answer found directly in the chunk
- Starts with a question word: What, How, When, Where, Why, or Who

Output ONLY a valid JSON object with exactly two keys: "question" and "answer".
Do NOT include markdown, explanations, or extra text.
If no valid question can be generated, output null.

Example: {"question": "When was Don Powell born?", "answer": "10 September 1946"}"""



print(f"Loading model: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()
print("✓ Model loaded.\n")

# ─────────────────────────────────────────────────────────────
# 4. HELPER: Generate unique chunk_id + parse LLM output
# ─────────────────────────────────────────────────────────────
def generate_chunk_id(metadata, line_number, text):
    """Create a unique, reproducible chunk_id from metadata + content."""
    # Primary: use article_id + line number (simple & traceable)
    article_id = metadata.get("article_id", "unknown")
    base_id = f"{article_id}_chunk_{line_number:05d}"
    
    # Fallback: hash the text if article_id missing
    if article_id == "unknown":
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        return f"chunk_{text_hash}_{line_number:05d}"
    
    return base_id

def parse_llm_response(response_text):
    """Extract {'question': ..., 'answer': ...} from LLM output."""
    cleaned = response_text.strip()
    # Remove markdown code blocks
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    
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

# ─────────────────────────────────────────────────────────────
# 5. MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"✗ Error: Input file not found: {INPUT_FILE}")
        return

    print(f"--- Starting generation from {INPUT_FILE} ---\n")
    processed = 0
    generated = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                chunk = json.loads(line)
                chunk_text = chunk.get("text", "").strip()
                metadata = chunk.get("metadata", {})

                # Filter by chunk length
                if len(chunk_text) < MIN_CHUNK_LEN or len(chunk_text) > MAX_CHUNK_LEN:
                    continue

                # Generate unique chunk_id
                chunk_id = generate_chunk_id(metadata, line_num, chunk_text)

                # Format prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Text chunk:\n{chunk_text}"}
                ]
                
                input_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Decode & parse
                response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                qa_pair = parse_llm_response(response)

                if qa_pair:
                    # Output: question + answer + metadata + chunk_id (NO chunk_text)
                    output_record = {
                        "chunk_id": chunk_id,                      # ← Key addition
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "source": {
                            "article_id": metadata.get("article_id"),
                            "title": metadata.get("title"),
                            "language": metadata.get("language"),
                            "url": metadata.get("url"),
                            "source_file": metadata.get("source_file")
                            # chunk_text intentionally omitted for lightweight output
                        }
                    }
                    outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    generated += 1

                processed += 1
                if processed % 50 == 0:
                    print(f"→ Processed {processed} chunks | Generated: {generated}")

            except Exception as e:
                print(f"✗ Error on line {line_num}: {type(e).__name__}: {e}")
                continue
            
            if processed % 20 == 0:
                torch.cuda.empty_cache()

    print(f"\n✅ Complete! Processed: {processed} | Generated: {generated}")
    print(f"   Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()