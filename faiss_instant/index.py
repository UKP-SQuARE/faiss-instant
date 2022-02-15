import argparse
import os
from unittest.result import failfast
import numpy as np
import torch
import pickle
from tqdm.auto import tqdm
import faiss
from . import info
import crash_ipdb


def load_embeddings(embeddings_dir):
    embedding_files = {}
    for fname in os.listdir(embeddings_dir):
        if 'embeddings.' in fname and '.pkl' in fname:
            chunk_number = int(fname.split('.')[1])
            fpath = os.path.join(embeddings_dir, fname)
            embedding_files[chunk_number] = fpath
    embeddings = []
    for i in tqdm(range(len(embedding_files))):
        with open(embedding_files[i], 'rb') as f:
            embs = pickle.load(f)
            embeddings.append(embs)
    embeddings = np.vstack(embeddings)
    print(f'Loaded embedding with shape: {embeddings.shape}')
    return embeddings

def set_nprobe(index, nprobe):
    changed = False
    if hasattr(index, 'nprobe') and nprobe and index.nprobe != nprobe:
        index.nprobe = nprobe
        print(f'Set nprobe = {nprobe}')
        changed = True
    return changed

def auto_nprobe(nlist):
    nprobe = min(max(round(2e-3 * nlist), 128), nlist)
    print(f"Automatic nprobe: {nprobe}")
    return nprobe

def auto_ivf_sq(nembeddings):
    # refering to https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    ncentroids = 0
    if nembeddings <= 1e6:
        ncentroids = int(np.ceil(16 * np.sqrt(nembeddings)))
    elif 1e6 < nembeddings <= 10e6:
        ncentroids = 65536
    elif 10e6 < nembeddings <= 100e6:
        ncentroids = 262144
    elif 100e6 <= nembeddings <= 1e9:
        ncentroids = 1048576
    else:
        raise ValueError('Too many embeddings! Please set the index factory string yourself')
    
    ncentroids = min(nembeddings // 39, ncentroids)
    index_factory_string = f'IVF{ncentroids},SQ4'
    print(f"Automatic index factory string: {index_factory_string}")
    return index_factory_string 

def run(embeddings_dir, output_dir, index_factory_string=None, distance='IP', nprobe=None):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ann.index')
    if not os.path.exists(output_path):
        using_gpu = torch.cuda.is_available()
        embeddings = load_embeddings(embeddings_dir)
        ndim = embeddings.shape[1]

        if index_factory_string is None:
            index_factory_string = auto_ivf_sq(len(embeddings))
        if distance == 'IP':
            distance = faiss.METRIC_INNER_PRODUCT
        elif distance == 'L2':
            distance = faiss.METRIC_L2
        else:
            raise NotImplementedError
        index = faiss.index_factory(ndim, index_factory_string, distance)
        
        if using_gpu:
            ngpus = faiss.get_num_gpus()
            print(f'Using {ngpus} gpus')
            index = faiss.index_cpu_to_all_gpus(index)
        
        index.train(embeddings)
        index.add(embeddings)
        if using_gpu:
            index = faiss.index_gpu_to_cpu(index)

        if nprobe is None and index.nlist:
            nprobe = auto_nprobe(index.nlist)
        set_nprobe(index, nprobe)
        faiss.write_index(index, output_path)
    else:
        print('Found existing index')
        index = faiss.read_index(output_path)
        if set_nprobe(index, nprobe):
            faiss.write_index(index, output_path)

    info.run(output_dir, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--index_factory_string', default=None, help='By default, it will use IVFSQ index')
    parser.add_argument('--distance', default='IP', choices=['L2', 'IP'])
    parser.add_argument('--nprobe', type=int, default=None, help='How many clusters you want to dive in for each search. By default it will be 2e-3 * nlist')
    args = parser.parse_args()
    run(**vars(args))