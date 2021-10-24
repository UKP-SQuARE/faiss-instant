import os
import faiss
import numpy as np

class FaissIndex(object):

    def __init__(self, resources_path):
        self.index_path = None
        self.ids_path = None
        for fname in os.listdir(resources_path):
            if fname.endswith('.txt'):
                self.ids_path = os.path.join(resources_path, fname)
            elif fname.endswith('.index'):
                self.index_path = os.path.join(resources_path, fname)
        assert self.index_path is not None, 'Cannot find any Faiss index with suffix .index'
        assert self.ids_path is not None, 'Cannot find any ID mapping file with suffix .txt'
        self.load()

    def load(self):
        self.index = faiss.read_index(self.index_path)
        self.pos2id = []
        with open(self.ids_path, 'r') as f:
            for line in f:
                self.pos2id.append(line.strip())
        
    def search(self, vectors, k):
        vectors = np.array(vectors, dtype=np.float32)
        scores, positions = self.index.search(vectors, k)
        scores = scores.tolist()
        ids = [map(lambda p: self.pos2id[p], positions_) for positions_ in positions]
        results = [dict(zip(ids_, scores_)) for scores_, ids_ in zip(scores, ids)]
        return results
