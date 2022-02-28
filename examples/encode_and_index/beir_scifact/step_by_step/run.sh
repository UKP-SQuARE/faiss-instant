export CUDA_VISIBLE_DEVICES=5,6

if [ ! -d "datasets/beir/scifact" ]; then
    mkdir -p datasets/beir
    cd datasets/beir
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
    unzip scifact.zip
fi

python -m faiss_instant.encode \
    --input_file datasets/beir/scifact/corpus.jsonl \
    --output_dir output \
    --model_type sbert \
    --model_name_or_path "sentence-transformers/msmarco-distilbert-base-tas-b" \
    --chunk_size 1600

python -m faiss_instant.index \
    --embeddings_dir output \
    --output_dir output

python -m faiss_instant.evaluate \
    --eval_dir "datasets/beir/scifact" \
    --model_name_or_path "sentence-transformers/msmarco-distilbert-base-tas-b" \
    --index_dir output \
    --output_dir output
