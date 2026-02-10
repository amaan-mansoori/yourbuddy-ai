from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("personal_knowledge.json", "r") as f:
    personal_data = json.load(f)

documents = []
for key, value in personal_data.items():
    if isinstance(value, list):
        for v in value:
            documents.append(f"{key}: {v}")
    else:
        documents.append(f"{key}: {value}")

embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def search(query, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [documents[i] for i in indices[0]]
