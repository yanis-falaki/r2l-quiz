from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from util import load_scifact_test, corpus_to_list

def dense_retrieval():
    # Retrieve scifact dataset
    corpus, queries, qrels = load_scifact_test()

    # Extract text from corpus into list
    c_text, c_ids = corpus_to_list(corpus)
    q_text = list(queries.values())
    q_ids = list(queries.keys())

    # Load pretrained embeddings model from Sentence Transformer model
    print("Loading embeddings model")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # encode corpus and queries
    print("Encoding corpus and queries")
    c_embeddings = model.encode(c_text)
    q_embeddings = model.encode(q_text)
    d = c_embeddings.shape[1]
    
    # Create index and store embeddings
    print("Building FAISS database")
    index = faiss.IndexFlatL2(d)
    index.add(c_embeddings)

    # Build results
    print("Building results json")
    k = 100
    D, I = index.search(q_embeddings, k)

    results = {}

    for qi, qid in enumerate(q_ids):
        results[qid] = {}
        for rank in range(k):
            cor_index = I[qi, rank]
            cor_id = c_ids[cor_index]

            # Negate L2 distance so that higher metric value is better. (Consistent with evaluation.py)
            score = float(-D[qi, rank])
            results[qid][cor_id] = score
    
    os.makedirs("results", exist_ok=True)
    with open("results/dense_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    dense_retrieval()