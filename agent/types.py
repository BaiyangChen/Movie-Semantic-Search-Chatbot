from pydantic import BaseModel, Field
from typing import Literal, Any

class VideoCitation(BaseModel):
    record_id: str
    start_sec: float
    end_sec: float
    chunk_file: str | None = None
    score: float | None = None

class AgentResponse(BaseModel):
    answer: str
    used_tools: list[str] = Field(default_factory=list)
    citations: list[VideoCitation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class ToolResult(BaseModel):
    name: str
    ok: bool
    content: Any = None
    error: str | None = None
    latency_ms: int | None = None

class VideoSearchResult(BaseModel):
    score: float
    faiss_id: int
    record_id: str
    start_sec: float
    end_sec: float
    chunk_file: str | None = None
    transcript: str | None = None
    visual_notes: str | None = None