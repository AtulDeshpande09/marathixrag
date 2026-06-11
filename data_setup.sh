#!/bin/bash
echo "creating folder"
mkdir data/
mkdir data/processed/
mkdir data/processed/questions
echo "downloading data"
cd data/processed/
wget https://huggingface.co/datasets/AtulDeshpande/marathiXragVectorDB/resolve/main/chunks_final.jsonl
echo "Chunks Download complete"

cd ..
cd ..
mkdir models

echo "downloading VectorDB..."
wget https://huggingface.co/datasets/AtulDeshpande/marathiXragVectorDB/resolve/main/chroma_db.zip
unzip chroma_db.zip
echo "Done!!!"