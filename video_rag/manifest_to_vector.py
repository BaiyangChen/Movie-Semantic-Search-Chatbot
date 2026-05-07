import argparse
import json
import numpy as np
import ollama
import faiss

from pathlib import Path


manifest_path = Path("data/video_index/visual_manifest.jsonl")
index_path = Path("data/video_index/faiss.index")
metadata_path = Path("data/video_index/faiss_metadata.json")

DEFAULT_EMBED_MODEL = "bge-m3:latest"

# jsonl to dict list
def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records

def make_embedded_text(record: dict) -> str:
    # Combine the text fields into a single string for embedding
    return f"""
    Transcript: {record.get('transcript', '')}\n
    visual_notes: {record.get('visual_notes', '')}
    """.strip()

def make_metadata(record):
    return {
        "record_id": record["id"],
        "start_sec": record["start_sec"],
        "end_sec": record["end_sec"],
        "chunk_file": record["chunk_file"],
        "transcript": record.get("transcript", ""),
        "visual_notes": record.get("visual_notes", ""),
    }

def main():
    records = read_jsonl(manifest_path)

    texts = []
    faiss_ids = []
    metadata_by_faiss_id = {}

    for i, record in enumerate(records):
        faiss_id = i + 1000
        texts.append(make_embedded_text(record))
        faiss_ids.append(faiss_id)
        metadata_by_faiss_id[faiss_id] = make_metadata(record)

    response = ollama.embed(model=DEFAULT_EMBED_MODEL, input=texts)

    vectors = np.array(response["embeddings"], dtype=np.float32)
    # L2 normalize the vectors before adding to the Faiss index
    faiss.normalize_L2(vectors)
    # All vectors should have the same dimension, so we can get it from the shape of the vectors array
    dimension = vectors.shape[1]
    # Create a Faiss index with the same dimension
    base_index = faiss.IndexFlatIP(dimension)
    # Wrap the base index with IndexIDMap, so we can further associate each vector with its corresponding ids
    index = faiss.IndexIDMap(base_index)
    # Add the vectors and their corresponding ids to the Faiss index
    index.add_with_ids(
        vectors,
        np.array(faiss_ids, dtype="int64"),
    )
    # Save the Faiss index
    faiss.write_index(index, str(index_path))

    metadata_file = {
        "embed_model": DEFAULT_EMBED_MODEL,
        "metadata_by_faiss_id": metadata_by_faiss_id,
    }
    
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_file, f, ensure_ascii=False, indent=2)
