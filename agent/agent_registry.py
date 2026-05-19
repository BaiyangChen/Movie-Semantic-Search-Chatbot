from typing import Dict, List
from agent.base_agent import BaseAgent
from agent.types import AgentResponse

import logging

class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tools: Dict[str, callable] = {}

    def register_agent(self, agent: BaseAgent, as_tool: bool = False) -> None:
        """Register an agent in the registry.

        Args:
            agent: An instance of a class that inherits from BaseAgent.
            as_tool: Whether to register the agent as a tool.
        """
        if agent.name in self.agents:
            logging.warning(f"Agent with name '{agent.name}' is already registered.")
        
        self.agents[agent.name] = agent

        if as_tool:
            tool = self.make_agent_tool(agent)
            self.tools[tool.__name__] = tool

    def get_agent_by_profile(self, profile: str) -> BaseAgent:
        """Get an agent based on a profile string.

        Args:
            profile: A string describing the agent's profile or capabilities.

        Returns:
            An instance of BaseAgent that matches the profile.
        """
        profile_map = {
            "Master": "master_agent",
            "VideoRAG": "video_agent",
        }
        agent_name = profile_map.get(profile, "master_agent")
        if not agent_name:
            raise ValueError(f"No agent found for profile '{profile}'.")
        return self.get_agent(agent_name)

    def get_agent(self, name: str) -> BaseAgent:

        if name not in self.agents:
            raise KeyError(f"No agent found with name '{name}'.")
        return self.agents[name]
    
    def make_agent_tool(self, agent: BaseAgent) -> callable:
        async def call_agent(query: str, chat_history: List[Dict] | None = None)-> AgentResponse:
            return await agent.run_without_stream(query=query, chat_history=chat_history, is_stream=False, can_think=False, temperature=0.7)
        
        call_agent.__name__ = f"{agent.name}_tool"
        call_agent.__doc__ = f"""
            {agent.description}
            
            Args:
                query: The user's question or input for the agent.
                
            returns:
                An AgentResponse object containing the agent's response, used tools, citations, and confidence level.
            """
        
        return call_agent
