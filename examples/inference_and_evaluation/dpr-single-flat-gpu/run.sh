# Installing packages and downloading data
if python -c "import dpr" &> /dev/null; then
    echo "Have installed DPR"
else
    git clone https://github.com/kwang2049/DPR
    cd DPR && pip install .
    cd ..
fi

if python -c "import beir" &> /dev/null; then
    echo "Have installed beir"
else
    pip install beir
fi

python -m dpr.data.download_data --resource data.retriever.qas.nq-test
python -m dpr.data.download_data --resource data.wikipedia_split.psgs_w100

# Downloading pre-computed index
if [ ! -d "index" ]; then
    mkdir index
    cd index
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/dpr-single-nq-base.size-full/nq-flat.txt
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/dpr-single-nq-base.size-full/nq-flat.index
    cd ..
fi


# Starting Faiss-instant with the index
export resources="${PWD}/index"
docker pull kwang2049/faiss-instant-gpu
docker run --runtime=nvidia --detach --rm -it -p 5001:5000 -v $resources:/opt/faiss-instant/resources --name faiss-instant-gpu kwang2049/faiss-instant-gpu  # Or `make run-gpu`
# docker run --runtime=nvidia --detach -e CUDA_VISIBLE_DEVICES=0,1,2 --rm -it -p 5001:5000 -v $resources:/opt/faiss-instant/resources --name faiss-instant-gpu kwang2049/faiss-instant-gpu  # Or `make run-gpu`


# Run the evaluation
python run.py --official_tokenization --batch_size 60 --device cuda


# Results:
# {
#     "ndcg": {
#         "NDCG@1": 0.45928,
#         "NDCG@2": 0.43299,
#         "NDCG@5": 0.40612,
#         "NDCG@10": 0.40415,
#         "NDCG@20": 0.42417,
#         "NDCG@50": 0.48221,
#         "NDCG@100": 0.55544
#     },
#     "map": {
#         "MAP@1": 0.10918,
#         "MAP@2": 0.16361,
#         "MAP@5": 0.22041,
#         "MAP@10": 0.25117,
#         "MAP@20": 0.27882,
#         "MAP@50": 0.31465,
#         "MAP@100": 0.34808
#     },
#     "acc.": {
#         "Accuracy@1": 0.45928,
#         "Accuracy@2": 0.56565,
#         "Accuracy@5": 0.68144,
#         "Accuracy@10": 0.74654,
#         "Accuracy@20": 0.79972,
#         "Accuracy@50": 0.83961,
#         "Accuracy@100": 0.85873
#     },
#     "precicion": {
#         "P@1": 0.45928,
#         "P@2": 0.40388,
#         "P@5": 0.29834,
#         "P@10": 0.22285,
#         "P@20": 0.16564,
#         "P@50": 0.11247,
#         "P@100": 0.08625
#     },
#     "mrr": {
#         "MRR@1": 0.45928,
#         "MRR@2": 0.51247,
#         "MRR@5": 0.54523,
#         "MRR@10": 0.55397,
#         "MRR@20": 0.55773,
#         "MRR@50": 0.55908,
#         "MRR@100": 0.55938
#     }
# }