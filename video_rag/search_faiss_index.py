import argparse
import json
import faiss
import numpy as np
import ollama

from agent.types import VideoSearchResult
from pathlib import Path

vector_path = Path("data/video_index/faiss.index")
metadata_path = Path("data/video_index/faiss_metadata.json")
model = "bge-m3:latest"

def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def embed_query(query: str, model: str) -> np.ndarray:
    response = ollama.embed(model=model, input=[query])
    
    # Faiss require Numpy array to be normalized for cosine similarity search
    vector = np.array(response["embeddings"], dtype=np.float32)
    faiss.normalize_L2(vector)
    return vector

def search_video_chunks(query: str, top_k: int = 5) -> list[dict]:
    # Load the Faiss index
    index = faiss.read_index(str(vector_path))
    metadata_file = load_metadata(metadata_path)

    query_vector = embed_query(query, model)

    scores, faiss_ids = index.search(query_vector, top_k)

    results = []

    paires = zip(scores[0], faiss_ids[0])
    for score, faiss_id in paires:
        if faiss_id == -1:
            continue
        
        metadata = metadata_file["metadata_by_faiss_id"].get(str(faiss_id), {})

        if metadata is None:
            continue

        results.append({
            "score": float(score),
            "faiss_id": int(faiss_id),
            "metadata": metadata
        })

    return results

def format_video_chunk_for_prompt(results: list[dict]) -> str:
    chunks = []

    for index, result in enumerate(results, start=1):
        chunk = f"""
            Retrieved chunk {index}
            score: {result["score"]:.4f}
            faiss_id: {result["faiss_id"]}
            record_id: {result["record_id"]}
            time: {result["start_sec"]}s - {result["end_sec"]}s
            chunk_file: {result["chunk_file"]}

            Transcript:
            {result.get("transcript") or "N/A"}

            Visual notes:
            {result.get("visual_notes") or "N/A"}
        """.strip()

        chunks.append(chunk)

    return "\n\n---\n\n".join(chunks)