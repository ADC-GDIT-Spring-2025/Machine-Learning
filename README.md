# Enron Email Dataset Retrieval Augmented Generation System

An advanced system for semantically processing and querying the Enron Email Dataset using vector databases, named entity recognition, enhanced chunking, and custom LLM integration.

## Project Overview

This project implements a Retrieval Augmented Generation (RAG) system optimized for email corpora like the Enron Email Dataset. It features:

- **Enhanced Semantic Chunking**: Text is chunked based on semantic coherence while preserving entity boundaries.
- **Named Entity Recognition (NER)**: GLiNER integration extracts and embeds entities into document metadata.
- **Sentence Overlap**: Maintains context continuity between chunks.
- **Vector Search with Qdrant**: Enables efficient cosine similarity and MMR search.
- **Multi-query Expansion**: Uses a prompt-driven approach to generate multiple search queries per user input.

## Installation

### Prerequisites

- Python 3.8+
- Pip

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd EnronEmailDataset
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the Enron email CSV file in the `data/` directory as `emails.csv`.

## Usage

### Processing and Indexing Emails

Emails are cleaned, metadata is extracted, and documents are chunked semantically:

```python
metadata, split_msg = extract_email_metadata(msg, idx)
documents = enhanced_chunker.create_documents(
    texts=[clean_text(msg)],
    metadatas=[metadata]
)
```

Vector storage is managed using Qdrant:

```python
db = Qdrant.from_existing_collection(modelemb, "emails_e5_qdrant", VECTOR_DB_NAME)
```

### Querying the System

Queries can be handled with semantic vector search using MMR and score-threshold approaches:

```python
retriever = db.as_retriever(search_kwargs={'k':20, 'search_type':'mmr'})
results = retrieval_chain_mmr.invoke({"input": "What did tung tung tung tung sahur tell baby gronk?"})
```

### Multi-query Retrieval

Multiple subqueries are generated for comprehensive search coverage:

```python
result = run_multi_query("What was discussed about mark-to-market accounting?")
print(result["final_answer"])
```

## Key Components

### EnhancedSemanticChunker

Extends LangChain's `SemanticChunker` with:

- NER-aware chunk boundaries using GLiNER
- Overlap handling at the sentence level
- Chunk enrichment with human-readable entity summaries

### Custom LLM Wrapper

Implements a LangChain-compatible wrapper over a remote LLaMA API with streaming and token tracking support.

### Retrieval Pipelines

- **Top-K Similarity Search**
- **MMR Re-ranking**
- **Score Threshold Search**

## Performance Tips

- Use GPU (`torch.cuda.is_available()`) for NER and embedding models.
- Cache intermediate chunking and embedding results to speed up reprocessing.

## File Structure

- `chunking.py` – Chunker logic
- `utilities.py` – Cleaning and metadata extraction
- `custom_llama_llm.py` – Custom LLM interface
- `initialize_groq.py` – Random API key loader (used for LLM)
- `data/` – Directory containing `emails.csv`

## License

MIT License

## Acknowledgments

- Enron Email Dataset
- GLiNER for Named Entity Recognition
- LangChain and Qdrant for RAG infrastructure
