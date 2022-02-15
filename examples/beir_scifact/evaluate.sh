python -m faiss_instant.evaluate \
    --eval_dir "datasets/beir/scifact" \
    --model_name_or_path "sentence-transformers/msmarco-distilbert-base-tas-b" \
    --index_dir output \
    --output_dir output

# {
#     "nDCG": {
#         "NDCG@1": 0.53,
#         "NDCG@3": 0.59392,
#         "NDCG@5": 0.61702,
#         "NDCG@10": 0.63966,
#         "NDCG@100": 0.6673,
#         "NDCG@1000": 0.67927
#     },
#     "MAP": {
#         "MAP@1": 0.50778,
#         "MAP@3": 0.56981,
#         "MAP@5": 0.58567,
#         "MAP@10": 0.59703,
#         "MAP@100": 0.60269,
#         "MAP@1000": 0.6031
#     },
#     "Recall": {
#         "Recall@1": 0.50778,
#         "Recall@3": 0.63983,
#         "Recall@5": 0.69283,
#         "Recall@10": 0.7565,
#         "Recall@100": 0.887,
#         "Recall@1000": 0.98333
#     },
#     "Precision": {
#         "P@1": 0.53,
#         "P@3": 0.22889,
#         "P@5": 0.15333,
#         "P@10": 0.08567,
#         "P@100": 0.01007,
#         "P@1000": 0.00111
#     },
#     "mrr": {
#         "MRR@1": 0.53,
#         "MRR@3": 0.58944,
#         "MRR@5": 0.60044,
#         "MRR@10": 0.60822,
#         "MRR@100": 0.61288,
#         "MRR@1000": 0.61323
#     },
#     "q/s (encoding)": 23.39,
#     "q/s (ANN search)": 14.92,
#     "q/s (total)": 9.11
# }