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
    async def run_without_stream(self, query: str, chat_history: list[dict] | None = None) -> AgentResponse:
        """Run the agent with the given query and optional chat history.

        Args:
            query: The user's question or input for the agent.
            chat_history: Optional list of previous messages in the conversation.
            config: Optional configuration for the agent run.

        Returns:
            An AgentResponse object containing the agent's response, used tools, citations, and confidence level.
        """
        pass

    @abstractmethod
    async def run_with_stream(self, query: str, chat_history: list[dict] | None = None) -> AgentResponse:
        """Run the agent in streaming mode with the given query and optional chat history.

        Args:
            query: The user's question or input for the agent.
            chat_history: Optional list of previous messages in the conversation.
            config: Optional configuration for the agent run.

        Returns:
            An asynchronous generator yielding partial AgentResponse objects as the agent processes the query.
        """
        pass