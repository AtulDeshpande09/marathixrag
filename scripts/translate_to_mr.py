import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sarvamai/sarvam-translate"
INPUT_FILE = "../data/processed/questions_EN.jsonl"   # From generate_questions.py
OUTPUT_FILE = "../data/processed/questions_MR.jsonl"  # Minimal Marathi output

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
model.eval()

def translate(text):
    """Simple English → Marathi translation."""
    messages = [
        {"role": "system", "content": "Translate to Marathi. Output only the translation."},
        {"role": "user", "content": text}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**model_inputs, max_new_tokens=256, temperature=0.01, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    result = output[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(result, skip_special_tokens=True).strip()

print(f"Translating {INPUT_FILE} → {OUTPUT_FILE}\n")

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    
    for i, line in enumerate(infile, 1):
        record = json.loads(line.strip())
        
        if "question" in record and "chunk_id" in record:
            print(f"[{i}] {record['chunk_id']}")
            
            # Minimal output: no English question, just Marathi + answer + metadata
            minimal_record = {
                "chunk_id": record["chunk_id"],              # ← Key to retrieve English later
                "marathi_question": translate(record["question"]),
                "answer": record["answer"],                  # Keep answer in English (dates/names)
                "source": record.get("source", {})           # Preserve metadata for sorting
            }
            
            outfile.write(json.dumps(minimal_record, ensure_ascii=False) + "\n")
            
            if i % 20 == 0:
                torch.cuda.empty_cache()

print(f"✅ Done! Minimal output saved to {OUTPUT_FILE}")