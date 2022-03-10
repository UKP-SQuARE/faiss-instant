export CUDA_VISIBLE_DEVICES=0  # Change this if you have multiple GPUs
# export CUDA_VISIBLE_DEVICES=5,6

if [ ! -d "datasets/beir/scifact" ]; then
    mkdir -p datasets/beir
    cd datasets/beir
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
    unzip scifact.zip
fi

python -m faiss_instant.encode_and_index \
    --input_file datasets/beir/scifact/corpus.jsonl \
    --output_dir output \
    --model_type sbert \
    --model_name_or_path "sentence-transformers/msmarco-distilbert-base-tas-b" \
    --chunk_size 1600