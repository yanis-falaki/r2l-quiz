# SciFact Retrieval: Sparse BM25 and Dense FAISS

This project compares two retrieval approaches on the BEIR SciFact test set:

- Sparse retrieval with BM25 using `rank-bm25`
- Dense retrieval with Sentence Transformers embeddings and a FAISS index

Both retrievers produce BEIR-style JSON result files that can be evaluated with the included evaluation script.

## Project Structure

```text
.
├── dense_retrieval.py              # Dense retriever using all-MiniLM-L6-v2 embeddings and FAISS
├── sparse_retrieval.py             # Sparse BM25 retriever
├── evaluation.py                   # Evaluates a result JSON file with BEIR metrics
├── util.py                         # Shared SciFact loading and corpus formatting helpers
├── requirements.txt                # Python dependencies
├── dense_retrieval_learning.ipynb  # Personal IPython Notebook for experimenting with libraries
├── results/                        # Generated retrieval outputs
│   ├── dense_results.json
│   └── sparse_results.json
└── datasets/                       # Downloaded SciFact data, created automatically
```

The retrieval scripts call `load_scifact_test()` from `util.py`, which downloads and unzips the SciFact dataset into `datasets/` if it is not already present.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Note: this environment uses `faiss-gpu`. If your machine does not have a compatible CUDA/GPU setup, replace it with the CPU FAISS package in `requirements.txt` before installing:

   ```text
   faiss-cpu
   ```

## Running Retrieval

Run the sparse BM25 retriever:

```bash
python sparse_retrieval.py
```

This creates:

```text
results/sparse_results.json
```

Run the dense FAISS retriever:

```bash
python dense_retrieval.py
```

This creates:

```text
results/dense_results.json
```

The dense script downloads the `sentence-transformers/all-MiniLM-L6-v2` model the first time it runs.

## Evaluating Results

Evaluate sparse retrieval:

```bash
python evaluation.py datasets/scifact results/sparse_results.json
```

Evaluate dense retrieval:

```bash
python evaluation.py datasets/scifact results/dense_results.json
```

The evaluation script reports NDCG, MAP, Recall, and Precision at 10 and 100.

## Results

On the SciFact test split, the saved result files produced these scores:

| Retriever | NDCG@10 | NDCG@100 | MAP@10 | MAP@100 | Recall@10 | Recall@100 | P@10 | P@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 sparse | 0.55970 | 0.58389 | 0.51473 | 0.52019 | 0.68617 | 0.79294 | 0.07633 | 0.00890 |
| Dense FAISS | 0.64508 | 0.67665 | 0.59593 | 0.60307 | 0.78333 | 0.92500 | 0.08833 | 0.01053 |

As expected, the dense retriever performed better across all reported metrics. This is likely due to the fact that the Sentence Transformer model can directly match semantic meaning between related claims and evidence documents, mitigating BM25's exact term overlap limitation. In SciFact, relevant scientific evidence may not use the exact wording described in the claim, which will favour dense embeddings which encode about semantics.

The trade-off is cost. BM25 is simpler, faster to build, and use less memory as it indexes term statistics rather than dense vectors. Conversely, dense retrieval requires loading a neural network to GPU, encoding every passage and query, then storing the passage embeddings in FAISS. It is more expensive in setup time and memory, however it gives higher quality results.