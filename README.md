# Faiss Instant
Build a Faiss service instantly. Faiss-instant will simply load **existing** Faiss index (and the corresponding ID mapping) and provide the search service via POST request. 

**New features:** Now Faiss-Instant also provides the toolkit for encoding texts into embeddings via [SBERT](https://sbert.net/) models and indexing the embeddings into a Faiss ANN index. One just needs to install the toolkit via
```bash
pip install faiss-instant
```
and try this [example](https://github.com/UKP-SQuARE/faiss-instant/blob/main/examples/beir_scifact/encode_and_index.sh).

## Usage
First, one needs to put the resource files (the ID mapping and the Faiss index, please refer to [resources/README.md](resources/README.md)) under the folder [./resources](./resources):
```bash
make download  # This will download example resource files. The example index comes from building a SQ index (QT_8bit_uniform) on a 10K-document version of the NQ corpus (dpr-single-nq-base was used for encoding). For other indices, please find under https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/.
```
Then, one needs to start the faiss-instant service via docker:
```bash
docker pull kwang2049/faiss-instant  # Or `make pull`; or `make build` to build the docker image
docker run --detach --rm -it -p 5001:5000 -v ${PWD}/resources:/opt/faiss-instant/resources --name faiss-instant kwang2049/faiss-instant  # Or `make run`; notice here a volume mapping will be made from ./resources to /opt/faiss-instant in the container
```
Finally, do the query:
```bash
bash query_example.sh  # curl 'localhost:5001/search' -X POST -d '{"k": 5, "vectors":  [[0.31800827383995056, -0.19993115961551666, -0.029884858056902885, ...]]}'
```
This will return the mappings from document IDs to the corresponding scores:
```json
[{"2426246":106.54305267333984,"4944584":107.05268096923828,"6195536":106.5833511352539,"6398884":107.19760131835938,"8077664":107.86164093017578}]
```
Whenever update the resources, one needs reload them:
```bash
curl 'localhost:5001/reload' -X GET  # Or `make reload`
```

## Advanced 
### Multiple Indices
One can have multiple indices in the resource folder, to load a certain one (actually a pair of `index_name`.index and `index_name`.txt, here the index name is 'ivf-32-sq-QT_8bit_uniform'):
```bash
curl -d '{"index_name":"ivf-32-sq-QT_8bit_uniform", "use_gpu":true}' -H "Content-Type: application/json" -X POST 'http://localhost:5001/reload'
```
To view the available indices under the resource folder and the index loaded, one can run:
```bash
curl -X GET 'http://localhost:5001/index_list'
```
To load a specified index:
```bash
curl -d '{"index_name":"ivf-32-sq-QT_8bit_uniform"}' -H "Content-Type: application/json" -X POST 'http://localhost:5001/reload'
```

### Use GPU
> Note Faiss only supports part of the index types: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#implemented-indexes. And for PQ, it cannot support large `m` such as 384.

One can also use GPU to accelerate the search. To achieve that, one needs to use the GPU version:
```bash
docker pull kwang2049/faiss-instant-gpu  # The current image supports only CUDA 10.2 or higher version
```
And then start the GPU-version container:
```bash
docker run --runtime=nvidia --detach --rm -it -p 5001:5000 -v ${PWD}/resources:/opt/faiss-instant/resources --name faiss-instant-gpu kwang2049/faiss-instant-gpu  # Or `make run-gpu`
```
This will load the index onto the 0th GPU by default. To load a specified index and make it on GPU, one can run:
```bash
curl -d '{"index_name":"ivf-32-sq-QT_8bit_uniform", "use_gpu":true}' -H "Content-Type: application/json" -X POST 'http://localhost:5001/reload'
```

### Reconstruct
To get the original vector without indexing by its ID, run:
```bash
curl -X 'GET' 'http://localhost:5001/reconstruct?id=1'  # This example returns the vector by its ID='1'
```

### Explain
To compute the similarity score between a given query vector and a support vector by its ID:
```bash
bash explain_example.sh
```


## Philosophy
Faiss-instant provides only the search service and relies on uploaded Faiss indices. By using the volume mapping, the huge pain of uploading index files to the docker service can be directly removed. Consequently, a minimal efficient Faiss system for search is born.

For creating index files (and also benchmarking ANN methods), please refer to [kwang2049/benchmarking-ann](https://github.com/kwang2049/benchmarking-ann).

## Reference
[plippe/faiss-web-service](https://github.com/plippe/faiss-web-service)
