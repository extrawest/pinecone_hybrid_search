#  Langchain Pinecone Hybrid Search Showcase 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Maintaner](https://img.shields.io/static/v1?label=Andriy%20Gulak&message=Maintainer&color=red)](mailto:andriy.gulak@extrawest.com)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/extrawest/pinecone_hybrid_search/issues)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## PROJECT INFO
- **Langchain**
- Pinecone for Vector Database
- HuggingFace all-MiniLM-L6-v2 for embeddings
- BM25 with mmh3 hashing encoder

## Features
- Hybrid Search is the combination of full text and vector queries that execute against a search index containing both searchable plain text content and generated embeddings

## Demo
```bash
Input sentences: ['In 2019, I visited Hungary', 'In 2020, I visited Czech Republic', 'In 2021, I visited Georgia']
Custom query: What country did I visit first?
100%|██████████| 3/3 [00:00<00:00, 24.12it/s]
BM25 values saved to bm25_values.json
100%|██████████| 1/1 [00:02<00:00,  2.15s/it]
Query result: [Document(metadata={'score': 0.286206543}, page_content='In 2019, I visited Hungary'), Document(metadata={'score': 0.255560637}, page_content='In 2020, I visited Czech Republic'), Document(metadata={'score': 0.225382119}, page_content='In 2021, I visited Georgia')]
```
**Generated bm25_values.json is present in the repo**

## Installing:
**1. Clone this repo to your folder:**

```
git clone THIS REPO
```

**2. Create a virtual environment**

**3. Install the dependencies**

```
pip install -r requirements.txt
``` 

[Extrawest.com](https://www.extrawest.com), 2024


