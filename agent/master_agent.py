import ollama

from agent.agent_prompt.prompt_template import PROMPTS
from agent.types import AgentResponse
from agent.agent_registry import AgentRegistry
from agent.base_agent import BaseAgent



class MasterAgent(BaseAgent):
    name = "master_agent"
    description = "The master agent is responsible for orchestrating the overall conversation flow and delegating tasks to specialized agents like the video_agent. It determines which agent to call based on the user's query and the context of the conversation. The master agent can also handle general questions that don't require specific tools or agents, and it can manage the chat history to provide context for follow-up questions."
    def __init__(self, model:str, agent_registry: AgentRegistry, temperature: float = 0.7):
        super().__init__()
        self.model = model
        self.agent_registry = agent_registry
        self.temperature = temperature

    async def run(self, query: str, chat_history: list[dict] | None = None) -> AgentResponse:
        used_tools = []
        tools = self.agent_registry.tools

        messages = [
            {
                "role": "system",
                "content": PROMPTS["master_agent"]
            },
        ]

        if chat_history:
            messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": query
        })

        for _ in range(5):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=list(tools.values()),
                stream=False,
                think=True,
                options={'temperature': self.temperature}
            )

            if response:
                assistant_message = response["message"]
                messages.append(assistant_message)
                tool_calls = assistant_message.get("tool_calls", [])

                if not tool_calls:
                    return AgentResponse(
                        answer=assistant_message["content"],
                        used_tools=used_tools,
                        citations=[],
                        confidence="medium"
                    )
                
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    if tool_name in self.agent_registry.tools:
                        used_tools.append(tool_name)
                        tool_func = self.agent_registry.tools[tool_name]
                        tool_response = await tool_func(**tool_args)

                        tool_response_content = tool_response.model_dump_json()

                        messages.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_response_content
                        })

                        break
                    else:
                        messages.append({
                            "role": "assistant",
                            "tool_name": tool_name,
                            "content": f"Error: Tool '{tool_name}' not found."
                        })
                        break

            