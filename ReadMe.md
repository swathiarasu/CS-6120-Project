# AcademEase: RAG System for Scientific Research Q&A

A Retrieval-Augmented Generation (RAG) application for answering questions from arXiv scientific papers using a hybrid approach that 
combines a FAISS-based index and optional live document upload. This project supports automated corpus download, indexing, and a fully 
containerized Streamlit interface.

## Folder Structure

```bash
├── app.py                         # Streamlit frontend for QA
├── build_index.py                # Builds FAISS index from arXiv metadata
├── download_corpus.py            # Downloads arXiv dataset from Kaggle
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker build configuration
├── entrypoints
│   └── run_app.sh                    # Script to launch the Streamlit app
│   └── build_index.sh                # Script to build FAISS index
│   └── download_corpus.sh           # Script to download the dataset
├── .kaggle/
│   └── kaggle.json               # Kaggle API credentials (place manually or mount via Docker)
├── data/
│   └── arxiv-metadata-oai-snapshot.json  # Downloaded dataset
│   └── index/
│       ├── faiss_index.bin       # FAISS vector index
│       └── metadata.pkl          # Metadata for the chunks
└── components/
    ├── rag_model.py              # RAG generation logic
    ├── retriever_multi.py        # Retriever using prebuilt FAISS index
    └── retriever_live.py         # Retriever for live-uploaded PDFs
```

## Steps to run this project

1. To spawn a container
   > docker build -t rag-app .

2. To downloading the corpus
   > GPU: `docker run --gpus all -v ./.kaggle:/root/.kaggle --entrypoint bash rag-app entrypoints/download_corpus.sh`
   >
   > CPU: `docker run -v ./.kaggle:/root/.kaggle --entrypoint bash rag-app entrypoints/download_corpus.sh`

3. To get the vector embeddings of the corpus
   > GPU: `docker run --gpus all --entrypoint bash rag-app entrypoints/build_index.sh`
   >
   > CPU: `docker run --entrypoint bash rag-app entrypoints/build_index.sh`

4. To run the app
   > GPU: `docker run -d --gpus all -p 8501:8501 rag-app`
   >
   > CPU: `docker run -d -p 8501:8501 rag-app`

5. To access the app, open your browser and go to: `http://<external-ip>:8501`