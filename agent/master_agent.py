import ollama
import json
import logging

from agent.agent_prompt.prompt_template import PROMPTS
from agent.types import AgentResponse
from agent.agent_registry import AgentRegistry
from agent.base_agent import BaseAgent
from tools import chainlit_helper



class MasterAgent(BaseAgent):
    name = "master_agent"
    description = "The master agent is responsible for orchestrating the overall conversation flow and delegating tasks to specialized agents like the video_agent. It determines which agent to call based on the user's query and the context of the conversation. The master agent can also handle general questions that don't require specific tools or agents, and it can manage the chat history to provide context for follow-up questions."
    def __init__(self, model:str, agent_registry: AgentRegistry):
        super().__init__()
        self.model = model
        self.agent_registry = agent_registry
        self.tools = self.agent_registry.tools

    async def handle_tool_calls(self, tool_calls:list[dict], messages: list[dict], used_tools:list[str]):
        final_tool_response = ""
    
        for tool_call in tool_calls:
            try:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                if tool_name in self.tools:
                    used_tools.append(tool_name)
                    tool_func = self.tools[tool_name]
                    tool_call_message = await tool_func(**tool_args)
                    final_tool_response = tool_call_message.answer
                    messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": final_tool_response
                            })
                else:
                    messages.append({
                                "role": "assistant",
                                "tool_name": tool_name,
                                "content": f"Error: Tool '{tool_name}' not found."
                            })
                    return
            except Exception as e:
                logging.exception("Master Agent Ollama call failed")
                return AgentResponse(
                    answer=f"LLM request failed: {str(e)}",
                    used_tools=used_tools,
                    citations=[],
                    confidence="low"
                )

    async def run_with_stream(self, query: str, chat_history: list[dict] | None = None, can_think: bool = False, is_stream: bool = False, temperature: float = 0.7):
        messages = chainlit_helper.build_message(query, self.name, chat_history)
        used_tools = []

        for _ in range(5):
            thinking_step = None
            tool_step = None
            tool_was_called = False

            stream = ollama.chat(
                model=self.model,
                messages=messages,
                tools=list(self.tools.values()),
                stream=is_stream,
                think=can_think,
                options={'temperature': temperature}
            )

            if can_think:
                thinking_step = chainlit_helper.create_step_for_chunk("Thinking", "llm")

            for chunk in stream:
                chunk_message = chunk.get("message", {})
                assistant_stream = chunk_message.get("content", "")
                tool_calls = chunk_message.get("tool_calls")
                think = chunk_message.get("thinking", "")

                if think and thinking_step is not None and can_think:
                    await chainlit_helper.think_step_with_stream(think, thinking_step)
                    await thinking_step.update()
                    continue

                if tool_calls:
                    tool_was_called = True
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                    })
                    if tool_step is None:
                        tool_step = chainlit_helper.create_step_for_chunk("Using Tool", "tool")

                    await chainlit_helper.use_tool_step(tool_calls[0], tool_step, is_stream)
                    await tool_step.send()

                    await self.handle_tool_calls(tool_calls, messages, used_tools)
                    continue

                if assistant_stream:
                    yield assistant_stream
            
            if not tool_was_called:
                return

    async def run_without_stream(self, query: str, chat_history: list[dict] | None = None, can_think: bool = False, is_stream: bool = False, temperature: float = 0.7):
        messages = chainlit_helper.build_message(query, self.name, chat_history)
        used_tools = []
        
        for _ in range(5):
            thinking_step = None
            tool_step = None
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=list(self.tools.values()),
                stream=is_stream,
                think=can_think,
                options={'temperature': temperature}
            )
            
            assistant_message = response["message"]
            think_message = assistant_message.get("thinking", "")
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls")

            if can_think and think_message:
                thinking_step = chainlit_helper.create_step_for_chunk("Thinking", "llm")
                await chainlit_helper.think_step_without_stream(think_message=think_message, thinking_step=thinking_step)
            
            if tool_step is None:
                tool_step = chainlit_helper.create_step_for_chunk("Using Tool","tool")

            if tool_calls:
                await chainlit_helper.use_tool_step(tool_calls[0], tool_step, is_stream)
                await self.handle_tool_calls(tool_calls=tool_calls, messages=messages, used_tools=used_tools)
                continue

            break

        return AgentResponse(
                answer=assistant_message["content"],
                used_tools=used_tools,
                citations=[],
                confidence="medium"
            )