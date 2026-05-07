import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# check if ffmpeg and ffprobe are available on PATH
def find_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"Could not find {name}. Please install ffmpeg and make sure {name} is on PATH."
        )
    return path

# use ffprobe to get video duration in seconds
def get_duration_seconds(video_path: Path, ffprobe_path: str) -> float:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

# covert seconds to HHMMSS format
def format_timecode(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}{minutes:02d}{secs:02d}"

# Find all video files in the input directory (recursively) and return a sorted list of their paths.
def iter_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

# Build a chunk plan based on the video duration, chunk length, and overlap, return list of (start, end) time tuples for each chunk.
def build_chunk_plan(duration: float, chunk_seconds: int, overlap_seconds: int) -> list[tuple[float, float]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than 0.")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds cannot be negative.")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds.")

    step = chunk_seconds - overlap_seconds
    count = max(1, math.ceil(duration / step))
    chunks = []

    for index in range(count):
        start = index * step
        if start >= duration:
            break
        end = min(start + chunk_seconds, duration)
        chunks.append((start, end))

    return chunks


def split_video(
        video_input_path: Path, 
        output_dir: Path, 
        ffmpeg_path: str, 
        ffprobe_path: str, 
        chunk_seconds: int, 
        overlap_seconds: int, 
        dry_run: bool
        ) -> list[dict]:
    duration = get_duration_seconds(video_input_path, ffprobe_path)
    chunk_plan = build_chunk_plan(duration, chunk_seconds, overlap_seconds)
    video_output_dir = output_dir / video_input_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    records = []

    for chunk_index, (start, end) in enumerate(chunk_plan):
        start_code = format_timecode(start)
        end_code = format_timecode(end)
        chunk_name = f"{video_input_path.stem}_{start_code}_{end_code}{video_input_path.suffix}"
        chunk_output_path = video_output_dir / chunk_name

        record = {
            "id": f"{video_input_path.stem}_{chunk_index:04d}_{start_code}_{end_code}",
            "video_file": str(video_input_path),
            "chunk_file": str(chunk_output_path),
            "chunk_index": chunk_index,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(end - start, 3),
        }
        records.append(record)

        if dry_run:
            print(f"Dry run: would split {video_input_path} from {start:.3f}s to {end:.3f}s into {chunk_output_path}")
            continue

        if chunk_output_path.exists():
            print(f"Chunk already exists, skipping: {chunk_output_path}")
            continue

        command = [
            ffmpeg_path,
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_input_path),
            "-t",
            str(end - start),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(chunk_output_path),
        ]

        subprocess.run(command, check=True)
        print(f"Created chunk: {chunk_output_path}")

    return records

# Write the list of chunk records to a JSONL manifest file.
def write_rag_manifest(records: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# use command line arguments to specify input directory, output directory, manifest path, chunk length, overlap, and dry run mode.
def parse_args():
    parser = argparse.ArgumentParser(description="Split videos into chunks for RAG.")
    parser.add_argument("--input-dir", default="data/videos", help="Directory containing source videos.")
    parser.add_argument("--output-dir", default="data/video_chunks", help="Directory for generated video chunks.")
    parser.add_argument(
        "--manifest",
        default="data/video_index/chunks_manifest.jsonl",
        help="JSONL file that records generated video chunks.",
    )
    parser.add_argument("--chunk-seconds", type=int, default=60, help="Length of each chunk in seconds.")
    parser.add_argument("--overlap-seconds", type=int, default=10, help="Overlap between chunks in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print the chunk plan without creating files.")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    ffmpeg_path = find_binary("ffmpeg")
    ffprobe_path = find_binary("ffprobe")
    videos = iter_video_files(input_dir)

    if not videos:
        print(f"No video files found in {input_dir}.")
        return
    
    all_records = []
    for video_path in videos:
        print(f"Processing {video_path}")
        records = split_video(
            video_path, 
            output_dir, 
            ffmpeg_path, 
            ffprobe_path, 
            args.chunk_seconds, 
            args.overlap_seconds, 
            args.dry_run
        )
        all_records.extend(records)

    if not args.dry_run:
        write_rag_manifest(all_records, manifest_path)
        print(f"Wrote manifest with {len(all_records)} chunks to {manifest_path}")
    else:
        print(f"Dry run complete. {len(all_records)} chunks would be created.")

if __name__ == "__main__":
    main()