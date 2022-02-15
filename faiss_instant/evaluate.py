import argparse
from collections import defaultdict
import json
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

from sentence_transformers import SentenceTransformer
import faiss

import time

from tqdm.auto import tqdm

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def run(eval_dir, model_name_or_path, index_dir, output_dir, split='test', k_values=[1,3,5,10,100,1000]):
    #### Provide the data_path where scifact has been downloaded and unzipped
    _, queries, qrels = GenericDataLoader(data_folder=eval_dir).load(split=split)

    sbert = SentenceTransformer(model_name_or_path)
    index = faiss.read_index(os.path.join(index_dir, 'ann.index'))

    with open(os.path.join(index_dir, 'ids.txt'), 'r') as f:
        ids = f.read().split('\n')

    results = defaultdict(dict)
    time_query_encoding = 0
    time_ann_search = 0
    for qid, query in tqdm(queries.items()):
        time_query_encoding_start = time.time()
        query_embedding = sbert.encode([query], show_progress_bar=False)
        time_query_encoding_end = time.time()
        time_query_encoding += time_query_encoding_end - time_query_encoding_start
        
        time_ann_search_start = time.time()
        _, I = index.search(query_embedding, 1000)
        time_ann_search_end = time.time()
        time_ann_search += time_ann_search_end - time_ann_search_start

        for j in range(I.shape[1]):
            pid = ids[I[0, j]]
            pseudo_score = 1 / (j + 1)
            results[qid][pid] = pseudo_score

    retriever = EvaluateRetrieval()
    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric='mrr')

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        metrics = {
            'nDCG': ndcg,
            'MAP': _map,
            'Recall': recall,
            'Precision': precision,
            'mrr': mrr,
            'q/s (encoding)': round(len(queries) / time_query_encoding, 2),
            'q/s (ANN search)': round(len(queries) / time_ann_search, 2),
            'q/s (total)': round(len(queries) / (time_query_encoding + time_ann_search) , 2)
        }
        json.dump(metrics, f, indent=4)

    print(f'{__name__}: Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', help='Path to the evaluation data folder')
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--index_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--split', choices=['train', 'test', 'dev'], default='test')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1,3,5,10,100,1000])
    args = parser.parse_args()
    run(**vars(args))

