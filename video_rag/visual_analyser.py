import argparse
import json
import shutil
import subprocess
import ollama
from pathlib import Path

DEFAULT_INPUT_MANIFEST = Path("data/video_index/transcripts_manifest.jsonl")
DEFAULT_OUTPUT_MANIFEST = Path("data/video_index/visual_manifest.jsonl")
DEFAULT_FRAMES_DIR = Path("data/frames")
model = "qwen3-vl:4b"

def read_transcript_manifest(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records.append(record)

    return records

def find_binary(name: str) -> str:
    path = shutil.which(name)

    if path is None:
        raise FileNotFoundError(f"Could not find {name}. Please install ffmpeg and make sure {name} is on PATH.")

    return path

def read_json(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records.append(record)

    return records

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok = True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

def choose_frame_time(duration_sec: float) -> list[float]:
    if duration_sec <= 0:
        return []
    
    if duration_sec <= 3:
        return [duration_sec / 2]
    
    frame_times = [duration_sec*0.1, duration_sec*0.5, duration_sec*0.9]

    safe_times = []
    for t in frame_times:
        t = max(0.5, t)
        t = min(duration_sec - 0.5, t)
        safe_times.append(round(t, 3))

    return sorted(set(safe_times))

def time_format(seconds: float)-> str:
    total_milliseconds = int(round(seconds * 1000))
    return f"{total_milliseconds:08d}ms"

def extract_frames(ffmpeg_path: Path, chunk_file: Path, video_time_sec: float, output_image: Path) -> None:
    output_image.parent.mkdir(parents = True, exist_ok = True)

    if output_image.exists():
        print(f"Frame already exists, skipping: {output_image}")
        return
    
    command = [
    ffmpeg_path,
    "-y",
    "-ss",
    str(video_time_sec),
    "-i",
    str(chunk_file),
    "-frames:v",
    "1",
    "-q:v",
    "2",
    str(output_image),
    ]

    subprocess.run(command, check=True)

def extract_frames_for_record(record: dict, frames_dir: Path, ffmpeg_path: Path) -> list[dict]:
    chunk_file = Path(record["chunk_file"])
    chunk_id = record["id"]
    duration_sec = record["duration_sec"]
    chunk_start_sec = record["start_sec"]

    frame_output_dir = frames_dir / chunk_id
    frame_times = choose_frame_time(duration_sec)

    frames = []

    for video_time_sec in frame_times:
        frame_time_code = time_format(video_time_sec)
        output_image_path = frame_output_dir / f"frame_{frame_time_code}.jpg"

        extract_frames(ffmpeg_path, chunk_file, video_time_sec, output_image_path)

        frames.append({
            "file" : str(output_image_path),
            "chunk_time_sec": video_time_sec,
            "video_time_sec": round(chunk_start_sec + video_time_sec, 3)
        })

    return frames

def analyze_frames(frames: list[dict], transcript: str) -> str:
    if not frames:
        return ""

    description = []

    for index, frame in enumerate(frames):
        image_path = Path(frame["file"])
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path

        if not image_path.exists():
            raise FileNotFoundError(f"Frame image does not exist: {image_path}")

        prompt = f"""
            This image is frame {index + 1} from a video chunk.

            Frame timing:
            chunk_time_sec={frame.get("chunk_time_sec")}, video_time_sec={frame.get("video_time_sec")}

            Transcript for the video chunk:
            {transcript}

            Describe what is visually visible in this single frame.
            Rules:
            - Focus on visible content from the image.
            - Use the transcript only as supporting context.
            - Do not claim something is visible unless it appears in the image.
            - Mention visible people, objects, scenes, UI, on-screen text, and actions.
            - Return one concise sentence.
        """

        response = ollama.generate(
            model=model,
            prompt=prompt,
            images=[image_path],
            think=False,
        )
        content = response.get("response", "").strip()

        if not content:
            raise RuntimeError(f"Ollama returned an empty response for frame {index + 1}: {response}")

        description.append(f"Frame {index + 1}: {content}")

    if not description:
        raise RuntimeError("Ollama returned no frame descriptions.")

    return "\n".join(description)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames and generate visual notes.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
        help="Path to transcripts_manifest.jsonl.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
        help="Path to visual_manifest.jsonl.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help="Directory where extracted frames will be saved.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    ffmpeg_path = find_binary("ffmpeg")
    records = read_json(args.input_manifest)
    output_records = []

    for record in records:
        print(f"Processing visual frames for: {record['id']}")

        frames = record.get("frames", [])

        if not frames:
            frames = extract_frames_for_record(record, args.frames_dir, ffmpeg_path)

        visual_notes = record.get("visual_notes", "")

        if not visual_notes:
            frame_descriptions = analyze_frames(frames, record.get("transcript", ""))
            if frame_descriptions:
                summary = ollama.generate(
                    model=model,
                    prompt=(
                        "Summarize what happens across these video frames in one concise paragraph.\n\n"
                        f"{frame_descriptions}"
                    ),
                    think=False,
                )
                record["visual_notes"] = summary.get("response", "").strip()
            else:
                record["visual_notes"] = frame_descriptions
        else:
            print("Visual notes already exist, skipping Ollama.")

        output_records.append(record)

    write_jsonl(output_records, args.output_manifest)
    print(f"Wrote visual manifest: {args.output_manifest}")

if __name__ == "__main__":
    main()
