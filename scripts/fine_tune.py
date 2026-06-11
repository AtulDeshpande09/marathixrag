import os
import json
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_ID = "BAAI/bge-m3"
TRAIN_FILE = "../data/experiment_splits/train_pairs_MR.jsonl"
OUTPUT_MODEL_DIR = "../models/bge-m3-ft-marathi"

def load_jsonl_to_dataset(file_path):
    """Loads JSONL train rows and formats them into HF Dataset keys."""
    queries = []
    positives = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                queries.append(data["query"])
                positives.append(data["positive"])
    
    # Sentence-Transformers v3 trainer explicitly requires 'anchor' and 'positive'
    return Dataset.from_dict({"anchor": queries, "positive": positives})

def main():
    if not os.path.exists(TRAIN_FILE):
        print(f"✗ Error: Training split file not found: {TRAIN_FILE}")
        return

    print("Step 1: Loading Marathi training pair datasets...")
    train_dataset = load_jsonl_to_dataset(TRAIN_FILE)
    print(f"✓ Successfully parsed {len(train_dataset)} train rows.")

    print("Step 2: Loading Base BGE-M3 model weights...")
    model = SentenceTransformer(MODEL_ID)
    
    # FIX 1: Set max sequence length on the model instance
    model.max_seq_length = 512  

    # FIX 2: Explicitly enable gradient checkpointing on the underlying PyTorch module
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # MultipleNegativesRankingLoss sets up infoNCE optimization across the hardware batch.
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("Step 3: Setting up training hyper-parameters...")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=3,                  
        per_device_train_batch_size=16,      # Lowered from 32 to 16 to drastically reduce baseline VRAM load
        learning_rate=2e-5,                  
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,                           
        
        # ─────────────────────────────────────────────────────────────
        # CRITICAL MEMORY OPTIMIZATIONS FOR 24GB VRAM
        # ─────────────────────────────────────────────────────────────
        gradient_checkpointing=True,         # Recomputes activations during backward pass to save ~60% VRAM
        dataloader_num_workers=2             # Smooths out data delivery from CPU to GPU
    )

    print("Step 4: Bootstrapping training framework loop...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    print("\n🚀 Starting Fine-Tuning Execution Profile...")
    trainer.train()

    print(f"\n✅ Training Complete! Exporting compiled weights asset files to: {OUTPUT_MODEL_DIR}")
    model.save_pretrained(OUTPUT_MODEL_DIR)

if __name__ == "__main__":
    main()
