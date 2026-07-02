from util import load_scifact_test, corpus_to_list
from rank_bm25 import BM25Okapi
import os
import json

def tokenize(text):
    return text.lower().split()

def sparse_retrieval():
    corpus, queries, qrels = load_scifact_test()

    c_text, c_ids = corpus_to_list(corpus)
    q_text = list(queries.values())
    q_ids = list(queries.keys())

    tokenized_corpus = [tokenize(doc) for doc in c_text]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    results = {}
    k = 100

    for qid, query, in zip(q_ids, q_text):
        tokenized_query = tokenize(query)

        scores = bm25.get_scores(tokenized_query)

        # get top-k document indices sorted in descending order
        top_k_indices = scores.argsort()[::-1][:k]

        results[qid] = {}
        for idx in top_k_indices:
            doc_id = c_ids[idx]
            score = float(scores[idx])
            results[qid][doc_id] = score

    os.makedirs("results", exist_ok=True)
    with open("results/sparse_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    sparse_retrieval()