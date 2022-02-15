from functools import partial
from typing import List
from torch import multiprocessing as mp
try:
     mp.set_start_method('spawn')  # needed by CUDA + multiprocessing
except RuntimeError:
    pass
import argparse
import pickle
import json
import linecache
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.util import batch_to_device
from tqdm.auto import tqdm
import crash_ipdb


class JSONLDataSet(Dataset):

    def __init__(self, data_path):
        assert os.path.exists(data_path)
        self.data_path = data_path
        self.nexamples = len(linecache.getlines(data_path))
    
    def __getitem__(self, index):
        if index >= self.nexamples:
            raise StopIteration

        line = linecache.getline(self.data_path, index + 1)
        line_dict = json.loads(line)
        text = ' '.join([line_dict['title'], line_dict['text']])
        doc_id = line_dict['_id']
        return InputExample(doc_id, [text])

    def __len__(self):
        return self.nexamples

def build_model(model_type, model_name_or_path):
    model_type = model_type.lower()
    if model_type in ['sbert']:
        return SentenceTransformer(model_name_or_path)
    else:
        # TODO: Add support to Transformer and Adapter models
        raise NotImplementedError

def save_chunk(chunk: List[torch.Tensor], output_dir, chunk_number):
    chunk = torch.cat(chunk).cpu().numpy()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'embeddings.{chunk_number}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(chunk, f)
    print(f'Dumped {len(chunk)} embeddings')

def smart_batching_collate(self, batch):
    """
    Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
    Here, batch is a list of tuples: [(tokens, label), ...]

    :param batch:
        a batch from a SmartBatchingDataset
    :return:
        a batch of tensors for the model
    """
    num_texts = len(batch[0].texts)
    texts = [[] for _ in range(num_texts)]
    ids = []

    for example in batch:
        for idx, text in enumerate(example.texts):
            texts[idx].append(text)

        ids.append(example.guid)

    sentence_features = []
    for idx in range(num_texts):
        tokenized = self.tokenize(texts[idx])
        batch_to_device(tokenized, self._target_device)
        sentence_features.append(tokenized)

    return sentence_features, ids

def run(input_file, output_dir, model_type, model_name_or_path, normalize=False, chunk_size=160000, batch_size_per_gpu=32):
    assert torch.cuda.is_available()
    ngpus = torch.cuda.device_count()
    batch_size = ngpus * batch_size_per_gpu

    dataset = JSONLDataSet(input_file)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=ngpus)
    
    model: SentenceTransformer = build_model(model_type, model_name_or_path)
    dataloader.collate_fn = partial(smart_batching_collate, model)
    model = DataParallel(model).cuda()

    chunk = []
    current_chunk_size = 0
    chunk_number = 0
    ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = batch[0][0]
            ids.extend(batch[1])
            embeddings = model(data)['sentence_embedding']
            if normalize:
                embeddings = F.normalize(embeddings, dim=-1)
            chunk.append(embeddings)
            current_chunk_size += len(embeddings)

            if current_chunk_size >= chunk_size:
                save_chunk(chunk, output_dir, chunk_number)
                
                chunk = []
                current_chunk_size = 0
                chunk_number += 1
    
    if len(chunk):
        save_chunk(chunk, output_dir, chunk_number)
        chunk_number += 1
    
    with open(os.path.join(output_dir, 'ids.txt'), 'w') as f:
        for line in ids:
            f.write(str(line) + '\n')

    print(f'Done: {chunk_number} chunks dumped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_dir')
    parser.add_argument('--model_type')
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--normalize', action='store_true', help='Set this flag if cosine-similarity will be used')
    parser.add_argument('--chunk_size', type=int, default=160000)
    parser.add_argument('--batch_size_per_gpu', type=int, default=32)
    args = parser.parse_args()
    run(**vars(args))

