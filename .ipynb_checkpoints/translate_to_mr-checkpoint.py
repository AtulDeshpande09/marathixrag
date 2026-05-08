import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup
model_id = "sarvamai/sarvam-translate"
INPUT_FILE = "generated_questions.jsonl"
FINAL_OUTPUT = "marathi_questions.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

def translate_to_marathi(text):
    # Official Sarvam-translate chat-style prompt
    messages = [
        {"role": "system", "content": "Translate the text below to Marathi. Preserve the 'Question (Answer)' format strictly."},
        {"role": "user", "content": text}
    ]

    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.01, # Keep it low for structural consistency
            num_return_sequences=1
        )
    
    # Slice to get only the generated part
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

# 2. Loop through your English JSONL
with open(INPUT_FILE, "r", encoding="utf-8") as in_f, \
     open(FINAL_OUTPUT, "w", encoding="utf-8") as out_f:
    
    for line in in_f:
        data = json.loads(line)
        eng_content = data.get("questions", "")
        
        if eng_content:
            print(f"Translating questions from {data['filename']}...")
            marathi_text = translate_to_marathi(eng_content)
            
            # Save both English and Marathi together
            data["questions_marathi"] = marathi_text
            out_f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Translation complete! Results saved in {FINAL_OUTPUT}")
