import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Configuration
model_id = "Qwen/Qwen2.5-7B-Instruct"

INPUT_DIRECTORIES = ["./data/raw/en", "./data/raw/hi"] 
OUTPUT_FILE = "generated_questions.jsonl"

# 2. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True
)

SYSTEM_PROMPT = """You will be given the content based on which you have to generate up to three challenging, logically coherent questions that strictly meet the following criteria:
1. Standalone & Additional Context Independent: The questions should be understandable without additional context and must not contain any references to “the paragraph” or “the article” outside of the content provided.
2. Unambiguous Answer: Each question should have a single, clear, and factual answer.
3. Grounded in Context & Conceptual Format: Each question must be conceptually rooted in the provided article’s content and follow this format:
 - Start with a clear question word (e.g., What, How, Where, When).
 - If no valid questions can be generated from the content, do not generate any questions.
For each question:
Provide the answer in parentheses after the question. The answer can be either one word or a phrase."""

# 3. Processing Loop
for folder in INPUT_DIRECTORIES:
    print(f"--- Starting Folder: {folder} ---")
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} not found. Skipping.")
        continue

    # Filter for JSON files
    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    for filename in files:
        file_path = os.path.join(folder, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                passages = data.get("passages", [])
                wiki_text = "\n\n".join(passages).strip()

            if not wiki_text:
                print(f"Skipping {filename}: No text found in 'passages'.")
                continue

            if len(wiki_text) > 400000:
                print(f"Warning: {filename} is very long. Truncating to stay within VRAM limits.")
                wiki_text = wiki_text[:400000]



            # Format Chat ML
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Content:\n{wiki_text}"}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

            # 4. Generate with memory management
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Strip the prompt tokens
            response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)

            # 5. Append results immediately
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                result = {
                    "source_dir": folder,
                    "filename": filename,
                    "questions": response.strip()
                }
                out_f.write(json.dumps(result) + "\n")

            print(f"Success: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
        
        # Periodically clear GPU cache for stability
        torch.cuda.empty_cache()

print("All processing complete.")

