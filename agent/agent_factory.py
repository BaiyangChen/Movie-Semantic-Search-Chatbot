from agent.agent_registry import AgentRegistry
from agent.master_agent import MasterAgent
from agent.video_agent import VideoAgent
from video_rag.video_index_service import VideoIndexService


def create_agent_registry(model: str) -> AgentRegistry:
    registry = AgentRegistry()

    video_agent = VideoAgent(
        model=model,
        index_service=VideoIndexService(),
    )
    master_agent = MasterAgent(
        model=model,
        agent_registry=registry,
    )

    registry.register_agent(video_agent, as_tool=True)
    registry.register_agent(master_agent, as_tool=False)

    return registry