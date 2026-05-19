from agent.types import VideoSearchResult
from video_rag.search_faiss_index import search_video_chunks


class VideoIndexService:
    def __init__(self, max_top_k: int = 10, score_threshold: float | None = None):
        self.max_top_k = max_top_k
        self.score_threshold = score_threshold

    def search(self, query: str, top_k: int = 5) -> list[VideoSearchResult]:
        query = query.strip()
        if not query:
            return []

        # The LLM may request a very small top_k. Keep enough context for
        # visually specific questions where the exact match can rank below
        # broad "stone/building/man" matches in pure vector search.
        top_k = min(max(top_k, 5), self.max_top_k)

        raw_results = search_video_chunks(query=query, top_k=top_k)

        results: list[VideoSearchResult] = []

        for item in raw_results:
            score = float(item.get("score", 0.0))

            if self.score_threshold is not None and score < self.score_threshold:
                continue

            metadata = item.get("metadata") or {}

            record_id = metadata.get("record_id")
            start_sec = metadata.get("start_sec")
            end_sec = metadata.get("end_sec")

            if record_id is None or start_sec is None or end_sec is None:
                continue

            results.append(
                VideoSearchResult(
                    score=score,
                    faiss_id=int(item.get("faiss_id", -1)),
                    record_id=str(record_id),
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    chunk_file=metadata.get("chunk_file"),
                    transcript=metadata.get("transcript") or "",
                    visual_notes=metadata.get("visual_notes") or "",
                )
            )

        return results
