import argparse
import json
import re
import faiss
import numpy as np
import ollama

from agent.types import VideoSearchResult
from pathlib import Path

vector_path = Path("data/video_index/faiss.index")
metadata_path = Path("data/video_index/faiss_metadata.json")
model = "bge-m3:latest"

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def embed_query(query: str, model: str) -> np.ndarray:
    response = ollama.embed(model=model, input=[query])
    
    # Faiss require Numpy array to be normalized for cosine similarity search
    vector = np.array(response["embeddings"], dtype=np.float32)
    faiss.normalize_L2(vector)
    return vector

def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]

def make_ngrams(tokens: list[str], size: int) -> set[str]:
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}

def make_metadata_text(metadata: dict) -> str:
    return " ".join([
        str(metadata.get("record_id") or ""),
        str(metadata.get("transcript") or ""),
        str(metadata.get("visual_notes") or ""),
    ])

def lexical_score(query: str, metadata: dict) -> float:
    """Small hybrid-search boost for exact visual descriptions in metadata."""
    query_tokens = [token for token in tokenize(query) if len(token) > 2]
    if not query_tokens:
        return 0.0

    metadata_text = make_metadata_text(metadata).lower()
    metadata_tokens = set(tokenize(metadata_text))
    query_token_set = set(query_tokens)

    token_overlap = len(query_token_set & metadata_tokens) / len(query_token_set)

    phrase_matches = 0
    for size in (2, 3, 4):
        for phrase in make_ngrams(query_tokens, size):
            if phrase in metadata_text:
                phrase_matches += size

    phrase_score = min(phrase_matches / max(len(query_tokens), 1), 1.0)

    start_bonus = 0.0
    query_lower = query.lower()
    if "begins with" in query_lower or "starts with" in query_lower:
        if "begins with" in metadata_text or "starts with" in metadata_text:
            start_bonus = 0.25

    return token_overlap + phrase_score + start_bonus

def search_video_chunks(query: str, top_k: int = 5) -> list[dict]:
    # Load the Faiss index
    index = faiss.read_index(str(vector_path))
    metadata_file = load_metadata(metadata_path)
    metadata_by_faiss_id = metadata_file["metadata_by_faiss_id"]

    query_vector = embed_query(query, model)

    candidate_k = min(int(index.ntotal), max(top_k, 20))
    scores, faiss_ids = index.search(query_vector, candidate_k)

    results_by_id = {}

    paires = zip(scores[0], faiss_ids[0])
    for score, faiss_id in paires:
        if faiss_id == -1:
            continue
        
        metadata = metadata_by_faiss_id.get(str(faiss_id), {})

        if metadata is None:
            continue

        lex_score = lexical_score(query, metadata)
        results_by_id[int(faiss_id)] = {
            "score": float(score) + lex_score,
            "faiss_id": int(faiss_id),
            "metadata": metadata
        }

    for faiss_id, metadata in metadata_by_faiss_id.items():
        lex_score = lexical_score(query, metadata)
        if lex_score < 0.55:
            continue

        faiss_id_int = int(faiss_id)
        existing = results_by_id.get(faiss_id_int)
        if existing is None or lex_score > existing["score"]:
            results_by_id[faiss_id_int] = {
                "score": lex_score,
                "faiss_id": faiss_id_int,
                "metadata": metadata
            }

    return sorted(results_by_id.values(), key=lambda item: item["score"], reverse=True)[:top_k]

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
