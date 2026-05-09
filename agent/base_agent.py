from abc import ABC, abstractmethod
from typing import Any, Dict
from agent.types import AgentResponse

class BaseAgent(ABC):
    name: str
    description: str

    def __init__(self):
        if not self.name or not self.description:
            raise ValueError("Agent must have a name and description")
        
    @abstractmethod
    async def run(self, query: str, chat_history: list[dict] | None = None) -> AgentResponse:
        """Run the agent with the given query and optional chat history.

        Args:
            query: The user's question or input for the agent.
            chat_history: Optional list of previous messages in the conversation.

        Returns:
            An AgentResponse object containing the agent's response, used tools, citations, and confidence level.
        """
        pass