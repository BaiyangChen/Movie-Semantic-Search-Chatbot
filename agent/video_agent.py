from video_rag.video_index_service import VideoIndexService
from video_rag.search_faiss_index import format_video_chunk_for_prompt
from agent.types import AgentResponse, VideoCitation
from agent.base_agent import BaseAgent
from tools import chainlit_helper
from typing import AsyncGenerator

import ollama
import json
import logging

class VideoAgent(BaseAgent):
    name = "video_agent"
    description = "An agent that answers questions about indexed video content, including scenes, timestamps, transcripts, and visual notes. It can search for relevant video chunks using the search_video_tool and use that information to answer user questions about the videos."
    def __init__(self, model: str, index_service: VideoIndexService):
        super().__init__()
        self.model = model
        self.index_service = index_service
    
    async def search_video_tool(self, query: str, top_k: int = 5) -> list[dict]:
        """Search indexed video chunks using the user's question.

            Args:
                query: The user's question about the indexed video content.
                top_k: The number of top results to return.

            Returns:
                A list of matching video chunks with timestamps, transcript, visual notes, and metadata.
        """
        results = self.index_service.search(query=query, top_k=top_k)
        return [r.model_dump() for r in results]
    
    async def handle_tool_calls(self, tool_calls: list[dict], message: list[dict], used_tools: list[str], citations: list[VideoCitation]):
        available_functions = {
            "search_video_tool": self.search_video_tool
        }

        for tool_call in tool_calls:
            try:
                tool_name = tool_call["function"]["name"]
                tool_args =  tool_call["function"]["arguments"]
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                if tool_name in available_functions:
                    used_tools.append(tool_name)
                    search_results = await available_functions[tool_name](**tool_args)

                    citations.extend(
                        VideoCitation(
                            record_id=item["record_id"],
                            start_sec=item["start_sec"],
                            end_sec=item["end_sec"],
                            chunk_file=item["chunk_file"],
                            score=item["score"],
                        ) for item in search_results
                    )

                    tool_response_content = format_video_chunk_for_prompt(search_results)

                    message.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": tool_response_content,
                    })
                else:
                    message.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Error: tool '{tool_name}' not found."
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

    async def run_without_stream(self, query: str, chat_history: list[dict] | None = None, can_think: bool = False, is_stream: bool = False, temperature: float = 0.7) -> AgentResponse:
        used_tools, citations, messages = chainlit_helper.init_run_state(query=query, chat_history=chat_history, agent_name=self.name)

        for _ in range(5):
            thinking_step = None
            tool_step = None

            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=[self.search_video_tool],
                stream=False,
                think=can_think,
                options={'temperature': temperature}
            )

            assistant_message = response["message"]
            think_message = assistant_message.get("thinking", "")
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", "")

            if can_think and think_message:
                thinking_step = chainlit_helper.create_step_for_chunk("Thinking", "llm")
                await chainlit_helper.think_step_without_stream(think_message=think_message, thinking_step=thinking_step)

            if tool_step is None:
                tool_step = chainlit_helper.create_step_for_chunk("Using Tool","tool")

            if tool_calls:
                await chainlit_helper.use_tool_step(tool_calls[0], tool_step, is_stream)
                await self.handle_tool_calls(tool_calls=tool_calls, message=messages, used_tools=used_tools, citations=citations)
                continue

            break

        return AgentResponse(
                answer=assistant_message["content"],
                used_tools=used_tools,
                citations=citations,
                confidence="medium"
            )
            

    async def run_with_stream(self, query: str, chat_history: list[dict] | None = None, can_think: bool = False, is_stream: bool = False, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        used_tools, citations, messages = chainlit_helper.init_run_state(query=query, chat_history=chat_history)
         
        for _ in range(5):
            thinking_step = None
            tool_step = None
            tool_was_called = False

            stream = ollama.chat(
                model=self.model,
                messages=messages,
                tools=[self.search_video_tool],
                stream=is_stream,
                think=can_think,
                options={'temperature': temperature}
            )

            if can_think:
                thinking_step = chainlit_helper.create_step_for_chunk("Thinking", "llm")
                await thinking_step.send()

            for chunk in stream:
                chunk_message = chunk.get("message", {})
                assistant_stream = chunk_message.get("content", "")
                tool_calls = chunk_message.get("tool_calls")
                think = chunk_message.get("thinking", "")
                
                if think and thinking_step is not None:
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
                        tool_step = chainlit_helper.create_step_for_chunk("Using Tool","tool")

                    await chainlit_helper.use_tool_step(tool_calls[0], tool_step, is_stream)
                    await tool_step.send()

                    await self.handle_tool_calls(tool_calls=tool_calls, message=messages, used_tools=used_tools, citations=citations)
                    continue
                
                if assistant_stream:
                    yield assistant_stream

            if not tool_was_called:
                return

                    

            
                
