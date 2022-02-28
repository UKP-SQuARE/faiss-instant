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
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/dpr-single-nq-base.size-full/nq-QT_8bit_uniform-ivf262144.index
    wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/dpr-single-nq-base.size-full/nq-QT_8bit_uniform-ivf262144.txt
    cd ..
fi

# Starting Faiss-instant with the index
export resources="$PWD/index"
docker pull kwang2049/faiss-instant  # Or `make pull`; or `make build` to build the docker image
docker run --detach --rm -it -p 5001:5000 -v $resources:/opt/faiss-instant/resources --name faiss-instant kwang2049/faiss-instant  # Or `make run`; notice here a volume mapping will be made from ./resources to /opt/faiss-instant in the container

# Run the evaluation
python run.py --official_tokenization


# Results:
# {
#     "ndcg": {
#         "NDCG@1": 0.45346,
#         "NDCG@2": 0.42814,
#         "NDCG@5": 0.40288,
#         "NDCG@10": 0.40197,
#         "NDCG@20": 0.42121,
#         "NDCG@50": 0.47772,
#         "NDCG@100": 0.54934
#     },
#     "map": {
#         "MAP@1": 0.11243,
#         "MAP@2": 0.1671,
#         "MAP@5": 0.22307,
#         "MAP@10": 0.25311,
#         "MAP@20": 0.27976,
#         "MAP@50": 0.31495,
#         "MAP@100": 0.34772
#     },
#     "acc.": {
#         "Accuracy@1": 0.45346,
#         "Accuracy@2": 0.55706,
#         "Accuracy@5": 0.67258,
#         "Accuracy@10": 0.73573,
#         "Accuracy@20": 0.7867,
#         "Accuracy@50": 0.82604,
#         "Accuracy@100": 0.84709
#     },
#     "precicion": {
#         "P@1": 0.45346,
#         "P@2": 0.39723,
#         "P@5": 0.29075,
#         "P@10": 0.21784,
#         "P@20": 0.16147,
#         "P@50": 0.1099,
#         "P@100": 0.08433
#     },
#     "mrr": {
#         "MRR@1": 0.45346,
#         "MRR@2": 0.50526,
#         "MRR@5": 0.53817,
#         "MRR@10": 0.54669,
#         "MRR@20": 0.5503,
#         "MRR@50": 0.55163,
#         "MRR@100": 0.55195
#     }
# }
