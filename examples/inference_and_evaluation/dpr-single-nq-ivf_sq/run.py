from collections import defaultdict
import csv
from email.policy import default
import json
import os
import sys
from time import sleep
from urllib import request
import numpy as np
import torch
from tqdm import tqdm
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import requests
from dpr.data.qa_validation import has_answer
from dpr.utils.tokenizers import SimpleTokenizer
from beir.retrieval.evaluation import EvaluateRetrieval
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--title', action='store_true', help='Use title or not for matching answers. By default, it is False, which is the same setting of the official DPR code.')
parser.add_argument('--official_tokenization', action='store_true', help='Whether to use the official tokenization (actually a wrongly written one). By default, it is False, whichi would give a little higher performance.')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Which device to use for query encoding')
args = parser.parse_args()

bsz = args.batch_size
use_title = args.title
device = args.device

ntries = 1000
faiss_container_working = False
for _ in range(ntries):
    try:
        response = requests.get('http://localhost:5001/index_list')
        if response.status_code == 200:
            faiss_container_working = True
            break
    except:
        pass
    sleep(1)

assert faiss_container_working, "Cannot reach the Faiss-instant container service!"


# 0. Loading test questions and answers
nq_test = {}
with open('downloads/data/retriever/qas/nq-test.csv', 'r') as f:
    reader = csv.reader(f, delimiter="\t")
    for qid, row in enumerate(reader):
        qid = str(qid)
        nq_test[qid] = {
            'question': row[0],
            'answers': eval(row[1])
        }

# 1. Retrieval
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
query_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
query_encoder.to(device)
query_encoder.eval()

@torch.no_grad()
def query(question):
    url = 'http://localhost:5001/search'
    inputs = tokenizer(
        question,
        add_special_tokens=True,
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    input_ids = inputs['input_ids']  # (1, seq_len)
    assert input_ids.shape[1] == 256
    input_ids[:, -1] = tokenizer.sep_token_id
    
    embeddings = query_encoder(input_ids).pooler_output.tolist()

    data = {"k": 100, "vectors": embeddings}
    data = json.dumps(data)
    response = requests.post(url, data)
    rels = json.loads(response.content)
    rels = [dict(sorted(rels_.items(), key=lambda x:x[1], reverse=True)) for rels_ in rels]
    return rels

fresults = f'retrieval_results.json'
if os.path.exists(fresults):
    with open(fresults, 'r') as f:
        retrieval_results = json.load(f)
else:
    retrieval_results = defaultdict(dict)
    questions = [v['question'] for k, v in nq_test.items()]
    qids = list(nq_test)
    for b in tqdm(range(0, len(questions), bsz)):
        questions_batch = questions[b:b+bsz]
        qids_batch = qids[b:b+bsz]
        rels_batch = query(questions_batch)
        for qid, rels in zip(qids_batch, rels_batch):
            retrieval_results[qid] = rels

    with open(fresults, 'w') as f:
        json.dump(retrieval_results, f)


# 2. Evaluation and calculating metrics
simple_tokenizer = SimpleTokenizer()
collection = {}
with open('downloads/data/wikipedia_split/psgs_w100.tsv', 'r') as f:
    f.readline()
    for line in tqdm(f, total=21e6):
        pid, passage, title = line.strip().split('\t')
        if use_title:
            collection[pid] = ' '.join([title, passage])
        else:
            collection[pid] = passage
acc = []
qrels = {}
for qid, question_answers in tqdm(nq_test.items()):
    answers = question_answers['answers']
    qrels[qid] = {}
    rels = retrieval_results[qid]
    contexts = {pid: collection[pid] for pid in rels}

    ha = 0
    for pid, context in contexts.items():
        if has_answer(answers, context, simple_tokenizer, 'string'):
            ha = 1
            qrels[qid][pid] = 1
        else:
            qrels[qid][pid] = 0
    
    acc.append(ha)


with open(f'qrels.use_title={use_title}.json', 'w') as f:
    json.dump(qrels, f)


k_values = [1, 2, 5, 10, 20, 50, 100]
assert len(qrels) == len(retrieval_results)
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, retrieval_results, k_values)
mrr = EvaluateRetrieval.evaluate_custom(qrels, retrieval_results, k_values, metric='mrr')
acc = EvaluateRetrieval.evaluate_custom(qrels, retrieval_results, k_values, metric='acc')
with open(f'metrics.use_title={use_title}.json', 'w') as f:
    json.dump({
        'ndcg': ndcg,
        'map': _map,
        'acc.': acc,
        'precicion': precision,
        'mrr': mrr
    }, f, indent=4)
