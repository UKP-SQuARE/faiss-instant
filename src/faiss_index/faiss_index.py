import os
import faiss
import re
import numpy as np
import GPUtil


class FaissIndex(object):
    ids_suffix = '.txt'
    index_suffix = '.index'

    def __init__(self, resources_path, logger, use_gpu=False):
        self.resources_path = resources_path
        self.index_loaded = None
        self.logger = logger
        try:
            self.load(use_gpu=use_gpu)
        except Exception as e:
            self.logger.error(e)

    @property
    def device(self):
        if hasattr(self.index, 'getDevice'):
            return 'gpu'
        else:
            return 'cpu'

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
        """
        Load a Faiss index and replace the current one in memory.
        :param index_name: Load specific index by its name. If left None, the first one will be loaded.
        :param use_gpu: Transfer the index into GPU. Only work for a certain list of index types due to Faiss' support.
        """
        index_list = self.parse_index_list()
        assert len(index_list), f"No valid index! Please make sure there is at least one pair called 'xxx{self.ids_suffix}' and 'xxx{self.index_suffix}' under the resource path."
        if index_name is None:
            index_name = index_list[0]
        assert index_name in index_list, f"Index '{index_name}' does not exist! Possible ones: {index_list}"
        
        ids_path = os.path.join(self.resources_path, f'{index_name}{self.ids_suffix}')
        index_path = os.path.join(self.resources_path, f'{index_name}{self.index_suffix}')
        self.logger.info(f'Loading index {index_name}')
        self.index = faiss.read_index(index_path)
        assert self.index.is_trained, 'The index has not been trained!'
        if use_gpu:
            assert type(self.index) in [faiss.IndexFlat, faiss.IndexIVFFlat, faiss.IndexIVFPQ, faiss.IndexIVFScalarQuantizer], 'The index type is not support for GPU by Faiss! More details: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#implemented-indexes.'
            if type(self.index) == faiss.IndexIVFPQ:
                assert self.index.pq.code_size in [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96], 'For IVFPQ with GPU, the code size is not supported by Faiss. More details: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#implemented-indexes.'
            assert len(GPUtil. getAvailable()), 'Cannot get access to GPUs!'
            res = faiss.StandardGpuResources()  # use a single GPU
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.pos2id = []
        self.id2pos = {}
        with open(ids_path, 'r') as f:
            for pos, line in enumerate(f):
                _id = line.strip()
                self.pos2id.append(_id)
                self.id2pos[_id] = pos
        
        self.index_loaded = index_name
        
    def search(self, qvecs, k):
        """
        Do the vector search.
        :param qvecs: The query vectors.
        :param k: Top-K items to return.
        :return: The list of top-K results for each query.
        """
        qvecs = np.array(qvecs, dtype=np.float32)
        scores, positions = self.index.search(qvecs, k)
        scores = scores.tolist()
        ids = [map(lambda p: self.pos2id[p], positions_) for positions_ in positions]
        results = [dict(zip(ids_, scores_)) for scores_, ids_ in zip(scores, ids)]
        return results

    def reconstruct(self, _id):
        """
        Reconstruct a vector from its corresponding ID.
        :param _id: The ID of the vector.
        :return: The single vector in shape (d,).
        """
        assert _id in self.id2pos, f'The ID {_id} is not included in the index!'
        pos = self.id2pos[_id]
        index = self.index
        try:
            index = faiss.index_gpu_to_cpu(index)  # GPU index does not support this method
        except:
            pass
        try:
            # Needed for IMI and IVF Indices: https://github.com/facebookresearch/faiss/issues/374#issuecomment-375019939.
            index.make_direct_map()
        except:
            pass
        return index.reconstruct(pos)
    
    def explain(self, qvec, _id):
        """
        Compute the similarity between the query vector and the support vector with a specific ID.
        :param qvec: The single query vector in shape (d,).
        :param _id: The ID of the vector.
        :return: The similarity score.
        """
        svec = self.reconstruct(_id)
        metric = self.index.metric_type
        assert metric in [faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2], 'This method only supports IP or L2 metrics.'
        if metric == faiss.METRIC_INNER_PRODUCT:
            score = np.dot(qvec, svec)
            m = 'faiss.METRIC_INNER_PRODUCT'
        if metric == faiss.METRIC_L2:
            score = np.linalg.norm(qvec - svec)
            m = 'faiss.METRIC_L2'
        result = {'score': score, 'metric': m}
        return result