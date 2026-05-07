import argparse
import json
from pathlib import Path
from faster_whisper import WhisperModel

INPUT_CHUNK_MANIFEST = Path("data/video_index/chunks_manifest.jsonl")
OUTPUT_TRANSCRIPT_MANIFEST = Path("data/video_index/transcripts_manifest.jsonl")

def load_model()->WhisperModel:
    model_size = "small"
    return WhisperModel(model_size, device="cpu", compute_type="int8")
    

def read_chunk_manifest(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records.append(record)

    return records

def write_transcript_manifest(records: list[dict], path:Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def transcribe_chunk(chunk_manifest_file: Path, model: WhisperModel, chunk_start_second: float) -> dict:
    segments, info = model.transcribe(str(chunk_manifest_file), beam_size=5)

    text_parts = []
    transcript_segments = []

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue
        
        text_parts.append(text)

        transcript_segments.append({
            "seg_start_sec": round(segment.start, 3),
            "seg_end_sec": round(segment.end, 3),
            "video_start_sec": round(chunk_start_second + segment.start, 3),
            "video_end_sec": round(chunk_start_second + segment.end, 3),
            "seg_text": text
        })

    return {
        "transcript": " ".join(text_parts).strip(),
        "transcript_segments": transcript_segments
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe video chunks into text.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=INPUT_CHUNK_MANIFEST,
        help="Path to chunks_manifest.jsonl.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=OUTPUT_TRANSCRIPT_MANIFEST,
        help="Path to transcripts_manifest.jsonl.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    records = read_chunk_manifest(args.input_manifest)
    output_records = []

    for record in records:
        chunk_file_path = Path(record["chunk_file"])
        chunk_start_second = float(record["start_sec"])
        model = load_model()

        result = transcribe_chunk(chunk_file_path, model, chunk_start_second)

        record["transcript"] = result["transcript"]
        record["transcript_segments"] = result["transcript_segments"]
        output_records.append(record)

        print(f"Transcribed: {chunk_file_path}")

    write_transcript_manifest(output_records, args.output_manifest)
    print(f"Wrote transcripts manifest: {args.output_manifest}")

if __name__ == "__main__":
    main()
