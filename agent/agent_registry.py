from typing import Dict, List
from agent.base_agent import BaseAgent
from agent.types import AgentResponse

class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tools: Dict[str, callable] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent in the registry.

        Args:
            agent: An instance of a class that inherits from BaseAgent.
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent with name '{agent.name}' is already registered.")
        
        self.agents[agent.name] = agent

        tool = self.make_agent_tool(agent)
        self.tools[tool.__name__] = tool

    def get_agent(self, name: str) -> BaseAgent:

        if name not in self.agents:
            raise KeyError(f"No agent found with name '{name}'.")
        return self.agents[name]
    
    def make_agent_tool(self, agent: BaseAgent) -> callable:
        async def call_agent(query: str, chat_history: List[Dict] | None = None)-> AgentResponse:
            return await agent.run(query=query, chat_history=chat_history)
        
        call_agent.__name__ = f"{agent.name}_tool"
        call_agent.__doc__ = f"""
            {agent.description}
            
            Args:
                query: The user's question or input for the agent.
                
            returns:
                An AgentResponse object containing the agent's response, used tools, citations, and confidence level.
            """
        
        return call_agent
