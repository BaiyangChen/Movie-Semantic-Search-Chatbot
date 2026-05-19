from tracemalloc import start
from agent.types import VideoCitation
from agent.base_agent import BaseAgent
from agent.agent_prompt.prompt_template import PROMPTS
from video_rag.search_faiss_index import format_video_chunk_for_prompt

import chainlit as cl
import logging
import time

from httpx import stream

thinking = False
start = time.time()

async def think_step_with_stream(think: str, thinking_step: cl.Step):
    if think:
        await thinking_step.stream_token(think)
        return True
    return False


def create_step_for_chunk(name, type) -> cl.Step:
    return cl.Step(name=name, type=type)


async def think_step_without_stream(think_message:str, thinking_step: cl.Step):
    if not think_message or not thinking_step:
        logging.warning("Thinking message or thinking step is missing.")
        return

    try:
        thinking_step.output = think_message
        await thinking_step.send()

    except Exception as e:
        logging.error("error in think_step_without_stream")

    return

async def use_tool_step(tool_call_dict: dict, tool_step: cl.Step, is_stream: bool):
    if not tool_call_dict or not tool_step:
        logging.warning("tool call dict or tool_step is missing")
        return

    try:
        tool_name = tool_call_dict['function']['name']
        if is_stream:
            await tool_step.stream_token(f"Tool called: {tool_name}")
        else:
            tool_step.output = f"Tool called: {tool_name}"
            await tool_step.send()
    except Exception as e:
        logging.error("error in use_tool_step")

def build_message(query: str, agent_name: str, chat_history: list[dict] | None = None,) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": PROMPTS[agent_name]
        },
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append({
        "role": "user",
        "content": query
    })

    return messages

        
def init_run_state(query: str, agent_name:str = "master_agent", chat_history: list[dict] | None = None) -> tuple[list[dict], list[str], list[VideoCitation]]:
    used_tools: list[str] = []
    citations: list[VideoCitation] = []
    messages = build_message(query=query, agent_name=agent_name, chat_history=chat_history)
    return used_tools, citations, messages
