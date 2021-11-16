import os
import faiss
import re
import numpy as np
import GPUtil


class FaissIndex(object):
    ids_suffix = '.txt'
    index_suffix = '.index'

    def __init__(self, resources_path, use_gpu=False):
        self.resources_path = resources_path
        self.index_loaded = None
        try:
            self.load(use_gpu=use_gpu)
        except Exception as e:
            print(e)

    def parse_index_list(self):
        ids_set = set()
        index_set = set()
        for fname in os.listdir(self.resources_path):
            if fname.endswith(self.ids_suffix):
                ids_set.add(re.sub(f'{re.escape(self.ids_suffix)}$', '', fname))
            if fname.endswith(self.index_suffix):
                index_set.add(re.sub(f'{re.escape(self.index_suffix)}$', '', fname))
        index_list = list(ids_set & index_set)
        return index_list

    def load(self, index_name=None, use_gpu=False):
        index_list = self.parse_index_list()
        assert len(index_list), f"No valid index! Please make sure there is at least one pair called 'xxx{self.ids_suffix}' and 'xxx{self.index_suffix}' under the resource path."
        if index_name is None:
            index_name = index_list[0]
        assert index_name in index_list, f"Index '{index_name}' does not exist! Possible ones: {index_list}"
        
        ids_path = os.path.join(self.resources_path, f'{index_name}{self.ids_suffix}')
        index_path = os.path.join(self.resources_path, f'{index_name}{self.index_suffix}')
        self.index = faiss.read_index(index_path)
        if use_gpu:
            print('Trying using GPUs. Please make sure the index type is supported by Faiss-GPU: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#implemented-indexes.')
            assert len(GPUtil. getAvailable()), 'Cannot get access to GPUs!'
            res = faiss.StandardGpuResources()  # use a single GPU
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.pos2id = []
        with open(ids_path, 'r') as f:
            for line in f:
                self.pos2id.append(line.strip())
        
        self.index_loaded = index_name
        
    def search(self, vectors, k):
        vectors = np.array(vectors, dtype=np.float32)
        scores, positions = self.index.search(vectors, k)
        scores = scores.tolist()
        ids = [map(lambda p: self.pos2id[p], positions_) for positions_ in positions]
        results = [dict(zip(ids_, scores_)) for scores_, ids_ in zip(scores, ids)]
        return results
