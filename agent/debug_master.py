import asyncio

from video_rag.video_index_service import VideoIndexService
from agent.video_agent import VideoAgent
from agent.agent_registry import AgentRegistry
from agent.master_agent import MasterAgent

MODEL = "qwen3-vl:4b"

async def main():
    index_service = VideoIndexService()

    video_agent = VideoAgent(
        model=MODEL,
        index_service=index_service,
        temperature=0.7,
    )

    registry = AgentRegistry()
    registry.register_agent(video_agent)

    master_agent = MasterAgent(
        model=MODEL,
        agent_registry=registry,
        temperature=0.7, 
    )

    result = await master_agent.run(
        query="In Attack on Titan episode 1, when does a titan first appear?"
    )

    print(result.model_dump_json(indent=2))

asyncio.run(main())