# MarathiXRetrieve : Cross-Lingual Information Retrieval (CLIR) Adaptation for Marathi-English RAG Systems

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License: MIT">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20RTX%203090-76B900?style=flat&logo=nvidia&logoColor=white" alt="GPU: RTX 3090">
  <br>
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat" alt="Status">
  <img src="https://img.shields.io/badge/Model-BGE--M3-blue?style=flat" alt="Model: BGE-M3">
  <img src="https://img.shields.io/badge/Task-Cross--Lingual%20IR-orange?style=flat" alt="Task: Cross-Lingual IR">
  <img src="https://img.shields.io/badge/Cost-%240.60%20USD-success?style=flat" alt="Cost: $0.60">
</p>

<p align="center">
  <b>An empirical study on mitigating the cross-lingual performance penalty in Information Retrieval.</b>
  <br>
  <i>Successfully cutting the cross-lingual retrieval error rate by 47.4% for Marathi-to-English document search.</i>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Step-by-Step Pipeline](#step-by-step-pipeline)
- [Model Weights & Datasets](#model-weights--datasets)
- [Evaluation Framework](#evaluation-framework)
- [Computational Efficiency](#computational-efficiency)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

Multilingual embedding models often exhibit a significant performance gap when retrieving cross-lingual documents (e.g., matching a Marathi query to an English document chunk) compared to monolingual English configurations. **MarathiXRetrieve** addresses this challenge by establishing a complete pipeline to:

1. **Synthetically generate** aligned evaluation datasets from Wikipedia articles
2. **Translate** English questions into Marathi using neural machine translation
3. **Fine-tune** `BAAI/bge-m3` using contrastive learning (`MultipleNegativesRankingLoss`)
4. **Evaluate** retrieval quality using a graded relevance framework

> **Core Hypothesis:** Targeted contrastive fine-tuning on synthetically aligned, article-aware low-resource language pairs can structurally reorganize the joint vector space to mitigate the cross-lingual penalty without inducing catastrophic forgetting in the baseline language.

---

## Key Results

| Metric | Base-Marathi | FT-Marathi | Delta |
|--------|-------------|------------|-------|
| **Top-1 Accuracy** | 0.7736 | **0.8020** | **+2.84%** |
| **Top-3 Accuracy** | 0.8378 | **0.8584** | +2.06% |
| **Top-5 Accuracy** | 0.8593 | **0.8744** | +1.51% |
| **Top-10 Accuracy** | 0.8758 | **0.8978** | +2.20% |
| **MRR** | 0.8099 | **0.8341** | +2.42% |
| **NDCG@5** | 0.6021 | **0.6372** | +3.51% |
| **NDCG@10** | 0.5268 | **0.5659** | +3.91% |
| **MAP** | 0.1391 | **0.1541** | +1.50% |

### Key Findings

- **Error Mitigation:** Cross-lingual Top-1 retrieval accuracy increased by **+2.84%** absolute (77.36% to 80.20%).
- **List Reorganization:** NDCG@10 improved by **+3.91%**, pushing relevant chunks higher in rankings.
- **No Catastrophic Forgetting:** Monolingual English performance changed by only **-0.92%**, confirming knowledge preservation.
- **Gap Compression:** The English-Marathi Top-1 accuracy gap compressed from **7.93% down to 4.17%**, a **47.4% reduction** in cross-lingual penalty.

---

## System Architecture

```
Wikipedia Articles
       |
       v
+---------------------+     +---------------------+     +---------------------+
|  1. Data Collection | --> |  2. Chunking        | --> |  3. Q&A Generation  |
|  (en + hi articles) |     |  (text preprocess)  |     |  (Qwen2.5-7B + vLLM)|
+---------------------+     +---------------------+     +---------------------+
                                                                  |
                                                                  v
+---------------------+     +---------------------+     +---------------------+
|  7. Evaluation      | <-- |  6. Fine-Tuning     | <-- |  4. Translation     |
|  (pytrec_eval)      |     |  (BGE-M3 + MNRL)    |     |  (sarvam-translate) |
+---------------------+     +---------------------+     +---------------------+
       ^                                                          |
       |                  +---------------------+                  |
       +------------------|  5. Vector Index    | <----------------+
                          |  (ChromaDB + BGE-M3)|
                          +---------------------+
```

### Core Architecture Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Synthetic Q&A Generation Engine** | `Qwen2.5-7B-Instruct-AWQ` on vLLM | High-throughput question generation with guided JSON output |
| **Cross-Lingual Translation Bridge** | `sarvamai/sarvam-translate` | Sentence-level English to Marathi alignment |
| **Vector Index Store** | `ChromaDB` | Native persistence with exact `chunk_id` index bindings |
| **Contrastive Optimization** | `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss` | Bi-encoder fine-tuning with gradient checkpointing |
| **Evaluation** | `pytrec_eval` | Standard TREC-style IR metrics with 3-tier graded relevance |

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (24GB VRAM recommended, e.g., NVIDIA RTX 3090/4090)
- [uv](https://github.com/astral-sh/uv) (recommended for fast package management)

### Setup

```bash
# Clone the repository
git clone https://github.com/AtulDeshpande09/marathixretrieve.git
cd marathixretrieve

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Fetch raw data from Hugging Face
bash data_setup.sh
```

### Requirements

```
torch
transformers
accelerate
bitsandbytes
chromadb
sentence-transformers
```

---

## Step-by-Step Pipeline

### 1. Build Document Chunks

Process raw Wikipedia articles into clean, filtered chunks:

```bash
uv run scripts/data_gen/build_chunks.py
```

### 2. Generate Synthetic Q&A Pairs

Generate English questions from document chunks using vLLM:

```bash
uv run scripts/data_gen/generate_questions.py
```

- **Model:** `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Output:** `data/processed/questions/questions_EN.jsonl`
- Uses `StructuredOutputsParams` with JSON schema for deterministic output

### 3. Translate to Marathi

Translate English questions to Marathi using sarvam-translate:

```bash
uv run scripts/data_gen/translate_to_mr.py
```

- **Model:** `sarvamai/sarvam-translate`
- **Output:** `data/processed/questions/questions_MR.jsonl`
- Processes in batches of 1,000 for efficiency

### 4. Generate Qrels (Relevance Judgments)

Create the graded relevance evaluation file:

```bash
uv run scripts/data_gen/generate_qrels.py
```

Uses a **3-tier graded relevance scale**:

| Grade | Label | Definition |
|-------|-------|------------|
| 2 | Exact Source Chunk | The exact chunk the question was generated from |
| 1 | Adjacent Sister Chunk | Another chunk from the same Wikipedia article |
| 0 | Irrelevant | Chunk from a different/unrelated article |

### 5. Create Embeddings & Index

Generate BGE-M3 embeddings and store in ChromaDB:

```bash
uv run scripts/data_gen/create_embeddings.py
```

### 6. Run Contrastive Fine-Tuning

Fine-tune BGE-M3 on Marathi-English query-document pairs:

```bash
uv run scripts/fine_tune.py
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Base Model | `BAAI/bge-m3` |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Max Seq Length | 512 |
| Gradient Checkpointing | Enabled |
| Mixed Precision | FP16 |
| Loss Function | `MultipleNegativesRankingLoss` |

### 7. Evaluate

Run the evaluation suite to compute retrieval metrics:

```bash
# Evaluate fine-tuned model
uv run scripts/evaluate/evaluate.py --model_path ./models/bge-m3-ft-marathi --mode fine_tuned

# Evaluate base model (for comparison)
uv run scripts/evaluate/evaluate.py --model_path BAAI/bge-m3 --mode base
```

---

## Model Weights & Datasets

To ensure full reproducibility, both the model weights and the generated synthetic evaluation datasets are openly hosted on the Hugging Face Hub:

<p align="center">
  <a href="https://huggingface.co/AtulDeshpande/bge-m3-ft-marathi">
    <img src="https://img.shields.io/badge/Model-AtulDeshpande/bge--m3--ft--marathi-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Model on HuggingFace">
  </a>
</p>

### Quick Inference Snippet

```python
from FlagEmbedding import BGEM3FlagModel

# Load the fine-tuned model directly from Hugging Face
model = BGEM3FlagModel('AtulDeshpande/bge-m3-ft-marathi', use_fp16=True)

# Encode cross-lingual Marathi queries
marathi_query = ["Don Powell चा जन्म केव्हा झाला?"]
embeddings = model.encode(marathi_query, batch_size=1, max_length=512)

# Use embeddings for similarity search against your English document index
```

### Gradio Demo

A demo notebook is available at `notebooks/Demo.ipynb` for interactive experimentation.

---

## Evaluation Framework

The evaluation uses standard IR metrics computed via `pytrec_eval`:

- **Top-K Accuracy** (K = 1, 3, 5, 10): Hit rate of any relevant document in top K
- **MRR (Mean Reciprocal Rank):** Position of the first relevant result
- **NDCG@K** (K = 5, 10): Graded relevance-aware ranking quality
- **MAP (Mean Average Precision):** Aggregate precision across recall levels

### Comprehensive Evaluation Matrix

| Metric | Base-English | FT-English | Base-Marathi | FT-Marathi | Delta (MR) |
|--------|-------------|------------|-------------|------------|------------|
| Top-1 Accuracy | 0.8529 | 0.8437 | 0.7736 | **0.8020** | +2.84% |
| Top-3 Accuracy | 0.8740 | 0.8813 | 0.8378 | **0.8584** | +2.06% |
| Top-5 Accuracy | 0.8841 | 0.8937 | 0.8593 | **0.8744** | +1.51% |
| Top-10 Accuracy | 0.9005 | 0.9083 | 0.8758 | **0.8978** | +2.20% |
| MRR | 0.8672 | 0.8652 | 0.8099 | **0.8341** | +2.42% |
| NDCG@5 | 0.7043 | 0.6995 | 0.6021 | **0.6372** | +3.51% |
| NDCG@10 | 0.6342 | 0.6274 | 0.5268 | **0.5659** | +3.91% |
| MAP | 0.1955 | 0.1885 | 0.1391 | **0.1541** | +1.50% |

---

## Computational Efficiency

An unexpected secondary outcome of this project was **extreme cost efficiency**, completing the entire pipeline for approximately **$0.60 USD**.

### Hardware & Optimizations

| Aspect | Detail |
|--------|--------|
| **GPU** | Single NVIDIA RTX 3090 (24GB VRAM) |
| **Generation** | vLLM with `guided_json` reduced 12,000 entity generation to under 2.5 hours |
| **VRAM Savings** | Gradient checkpointing + seq length 512 = ~60% memory reduction |
| **Training Stability** | FP16 mixed precision, batch size 16, 3 epochs |
| **Total Cost** | ~$0.60 USD |

---

## Repository Structure

```
marathixretrieve/
├── data/
│   ├── raw/                          # Wikipedia articles (en + hi)
│   ├── processed/
│   │   ├── chunks_final.jsonl        # Filtered document chunks
│   │   └── questions/
│   │       ├── questions_EN.jsonl    # Generated English questions
│   │       └── questions_MR.jsonl    # Translated Marathi questions
│   └── experiment_splits/
│       ├── train_pairs_MR.jsonl      # 80% training pairs
│       ├── eval_queries_MR.json      # 20% evaluation queries
│       └── qrels_MR.tsv              # Multi-graded TREC relevance judgments
├── experiments/                      # Experiment logs and results
├── notebooks/
│   └── Demo.ipynb                    # Interactive Gradio demo
├── sample_data/                      # Sample data for quick testing
├── scripts/
│   ├── data_gen/
│   │   ├── build_chunks.py           # Chunk builder
│   │   ├── chunk_id.py               # Chunk ID assignment
│   │   ├── create_embeddings.py      # Embedding generation
│   │   ├── generate_qrels.py         # Qrels file generation
│   │   ├── generate_questions.py     # vLLM Q&A generation
│   │   ├── get_wiki_articles.py      # Wikipedia data fetcher
│   │   └── translate_to_mr.py        # Marathi translation
│   ├── evaluate/
│   │   ├── evaluate.py               # Multi-K metric computation
│   │   ├── evaluate_top_k.py         # Top-K evaluator
│   │   └── logger.py                 # Experiment logger
│   ├── analyze_chunk_ratio.py        # Chunk statistics
│   └── fine_tune.py                  # Contrastive fine-tuning
├── .gitignore
├── data_setup.sh                     # One-click data setup
├── generated_questions.jsonl         # Sample generated questions
├── marathi_questions.jsonl           # Sample Marathi translations
├── README.md
└── requirements.txt
```

---

## Contributing

Contributions are welcome! If you'd like to extend this work to other low-resource languages or improve the pipeline:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Ideas for Extension

- [ ] Hard negative mining during training
- [ ] Evaluation on additional Marathi domains beyond Wikipedia
- [ ] Temperature-scaled InfoNCE loss exploration
- [ ] Scaling to larger synthetic datasets
- [ ] Support for additional low-resource language pairs

---

## Citation

If you use this work in your research, please cite:

Technical Report : [Zenodo](https://doi.org/10.5281/zenodo.20789161)

```bibtex
@software{deshpande2026marathixretrieve,
  author = {Deshpande, Atul},
  title = {MarathiXRetrieve: Cross-Lingual Information Retrieval Adaptation for Marathi-English RAG Systems},
  year = {2026},
  url = {https://github.com/AtulDeshpande09/marathixretrieve}
}
```

### Acknowledgments

- **BAAI** for the [bge-m3](https://huggingface.co/BAAI/bge-m3) model
- **Sarvam AI** for the [sarvam-translate](https://huggingface.co/sarvamai/sarvam-translate) model
- **Qwen Team** for [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Wikipedia** and **Hugging Face** for open datasets and model hosting infrastructure

---

<p align="center">
  <sub>Built with ❤️ for the Marathi language community.</sub>
  <br>
  <sub><a href="https://github.com/AtulDeshpande09/marathixretrieve">github.com/AtulDeshpande09/marathixretrieve</a></sub>
</p>
