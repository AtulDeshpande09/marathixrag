# MarathiXRetrieve : Cross-Lingual Information Retrieval (CLIR) Adaptation for Marathi-English RAG Systems

[![Python 3.12](https://shields.io)](https://python.org)
[![vLLM](https://shields.io)](https://github.com/vllm-project/vllm)
[![Hugging Face](https://shields.io)](https://huggingface.co)

An empirical study on mitigating the cross-lingual performance penalty in Information Retrieval. This project establishes a pipeline to synthetically generate aligned evaluation datasets and apply contrastive fine-tuning to **BAAI/bge-m3** using `MultipleNegativesRankingLoss`, successfully cutting the cross-lingual retrieval error rate nearly in half for Marathi-to-English document search.

---

## 🔬 Experimental Overview & Hypothesis

Multilingual embedding models often exhibit a significant performance gap when retrieving cross-lingual documents (e.g., matching a Marathi query to an English document chunk) compared to monolingual English configurations. 

**Core Hypothesis:** Targeted contrastive fine-tuning on synthetically aligned, article-aware low-resource language pairs can structurally reorganize the joint vector space to mitigate the cross-lingual penalty without inducing catastrophic forgetting in the baseline language.

### Core Architecture Components
* **Synthetic Q&A Generation Engine:** `Qwen2.5-7B-Instruct-AWQ` running on a high-throughput `vLLM` orchestration layer.
* **Cross-Lingual Translation Bridge:** `sarvamai/sarvam-translate` executing sentence-level alignment.
* **Vector Index Store:** `ChromaDB` native persistence layer using exact `chunk_id` index key bindings.
* **Contrastive Optimization:** Bi-Encoder fine-tuning using `SentenceTransformerTrainer` optimized with Gradient Checkpointing.

---

## 📈 Empirical Evaluation & Results

Retrieval performance was evaluated using standard Information Retrieval (IR) metrics calculated via `pytrec_eval`. The evaluation uses a **3-tier graded relevance scale** (Grade 2: Exact Source Chunk; Grade 1: Adjacent Sister Chunks from the same master document; Grade 0: Irrelevant data).

### Comprehensive Evaluation Matrix

| Evaluation Metric | Base-English (Zero-Shot) | FT-English (Sanity Check) | Base-Marathi (Zero-Shot) | FT-Marathi (Main Result) | Delta (Marathi) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 0.8529 | 0.8437 | 0.7736 | **0.8020** | **+2.84%** |
| **Top-3 Accuracy** | 0.8740 | 0.8813 | 0.8378 | **0.8584** | **+2.06%** |
| **Top-5 Accuracy** | 0.8841 | 0.8937 | 0.8593 | **0.8744** | **+1.51%** |
| **Top-10 Accuracy**| 0.9005 | 0.9083 | 0.8758 | **0.8978** | **+2.20%** |
| **MRR**            | 0.8672 | 0.8652 | 0.8099 | **0.8341** | **+2.42%** |
| **NDCG@5**         | 0.7043 | 0.6995 | 0.6021 | **0.6372** | **+3.51%** |
| **NDCG@10**        | 0.6342 | 0.6274 | 0.5268 | **0.5659** | **+3.91%** |
| **MAP**            | 0.1955 | 0.1885 | 0.1391 | **0.1541** | **+1.50%** |

### Key Findings
1. **Error Mitigation:** Cross-lingual Top-1 retrieval accuracy for Marathi query streams increased by **+2.84%** absolute, scaling from 77.36% to 80.20%.
2. **List Reorganization:** NDCG@10 improved by **+3.91%**, indicating the model successfully pushed highly-relevant and adjacent context chunks up the candidate ranking array.
3. **Preservation of Baseline Knowledge:** Monolingual English retrieval (`Base-English` vs `FT-English`) experienced a negligible variance of only **-0.92%**, confirming that conservative learning rates (\(2\times10^{-5}\)) prevent catastrophic forgetting during single-language adaptation sweeps.
4. **Closing the Discrepancy Gap:** The performance gap between English and Marathi Top-1 retrieval was compressed from **7.93% down to 4.17%**—effectively reducing the cross-lingual penalty by **47.4%**.

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/
│   │   └── en/
│   │   |   ├── article_1.json      
│   │   |   └── article_2.json
│   │   |   └── article_n.json
│   │   └── hi/
│   │       ├── article_1.json      
│   │       └── article_2.json
│   │       └── article_n.json      
│   ├── processed/
│   │   ├── chunks_final.jsonl          # Raw document source text chunks
│   │   └── questions/
│   │       ├── questions_EN.jsonl      # Quantized vLLM structured Qwen outputs
│   │       └── questions_MR.jsonl      # Paired Marathi translation questions
│   └── experiment_splits/
│       ├── train_pairs_MR.jsonl        # 80% Fine-Tuning dataset entries
│       ├── eval_queries_MR.json        # 20% Unseen evaluation queries
│       └── qrels_MR.tsv                # Standard multi-graded TREC answer keys
├── scripts/
│   ├── data_gen/
│   │   ├── generate_questions.py       # High-throughput vLLM generation script
│   │   └── translate_to_mr.py          # Translation alignment engine script
│   │   └── generate_qrels.py          # Qrels file generation script
|   |
│   ├── fine_tune.py
│   │   
│   └── evaluate/
│       ├── logger.py                   # SentenceTransformer contrastive trainer
│       └── evaluate.py                 # Multi-K metric computation script
└── data_setup.sh                       # Bash script for fetching data from HF
```

---

## 🛠️ Step-by-Step Pipeline Execution

### 1. Environment Initialization
This environment requires modern package resolution hooks capable of handling pre-compiled C++ extensions natively on GPU instances (e.g., NVIDIA RTX 3090/4090).

```bash
# Clone the repository
git clone https://github.com/AtulDeshpande09/marathixretrieve
cd marathixretrieve

# Install unified dependency stack using uv or standard pip
uv pip install -r requirements.txt
```

### 2. High-Throughput Synthetic Data Generation
Generates evaluation targets using hardware-accelerated batching and structured JSON validation via vLLM.
```bash
uv run scripts/data_gen/generate_questions.py
uv run scripts/data_gen/translate_to_mr.py
```

### 3. Smart Dataset Partitioning
Segments the dataset into 80/20 train/test structures and automatically maps multi-graded hierarchical scores (Grade 2 for primary context blocks, Grade 1 for sister components belonging to the same root document ID).
```bash
uv run scripts/data_gen/generate_qrels.py
```

### 4. Vector Database Index Setup
Initializes and indexes raw textual chunks directly to a clean database structure, isolated by strict ID mapping coordinates.
```bash
uv run scripts/data_gen/create_embeddings.py
```

### 5. Running the Baseline Benchmarks
Evaluates the original un-trained weights across multiple cutoffs ($K=[1,3,5,10]$) to freeze the initial system state.
```bash
uv run scripts/evaluate/evaluate.py --model_path BAAI/bge-m3 --mode baseline
```

### 6. Contrastive Fine-Tuning Configuration
Executes contractive weight updates. Optimized using gradient checkpointing to fully fit parameters within standard server-tier VRAM footprints (24GB).
```bash
uv run scripts/fine_tune.py
```

### 7. Final Model Evaluation
Evaluates the updated weights to produce the final comparative metrics matrix.
```bash
uv run scripts/evaluate/evaluate.py --model_path ./models/bge-m3-ft-marathi --mode fine_tuned
```

---

## ⚡ Computational & Resource Optimization

An unexpected secondary outcome of this project was extreme cost efficiency, completing the entire generation, translation, database indexing, and model training loop for a total infrastructure cost of **~$0.60 USD**.

* **Hardware Used:** Single Instance NVIDIA RTX 3090 (24GB VRAM).
* **vLLM Acceleration:** Reduced generation runtime to under 2.5 hours for 12,000 entities by passing native schema definitions directly via `guided_json`, preventing format parsing retry penalties.
* **VRAM Constraints Mitigated:** Resolved training-phase memory allocations by shrinking maximum sequence parameters to 512 context blocks and configuring `gradient_checkpointing=True`, reducing memory requirements by approximately 60% and keeping the workflow stable inside server VRAM boundaries.


* Sarvam AI
* Hugging Face
* Wikipedia
* BAAI

---

## Demo (Gradio) :



https://github.com/user-attachments/assets/280361bf-bc93-4490-9fef-6e7b20884481

