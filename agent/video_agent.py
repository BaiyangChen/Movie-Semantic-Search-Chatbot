from video_rag.video_index_service import VideoIndexService
from video_rag.search_faiss_index import format_video_chunk_for_prompt
from agent.types import AgentResponse, VideoCitation
from agent.base_agent import BaseAgent

from agent.agent_prompt.prompt_template import PROMPTS
import ollama


class VideoAgent(BaseAgent):
    name = "video_agent"
    description = "An agent that answers questions about indexed video content, including scenes, timestamps, transcripts, and visual notes. It can search for relevant video chunks using the search_video_tool and use that information to answer user questions about the videos."
    def __init__(self, model: str, index_service: VideoIndexService, temperature: float = 0.7):
        super().__init__()
        self.model = model
        self.index_service = index_service
        self.temperature = temperature

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

    async def run(self, query: str, chat_history: list[dict] | None = None) -> AgentResponse:
        used_tools = []
        citations = []
        available_functions = {
            "search_video_tool": self.search_video_tool
        }
        messages = [
            {
                "role": "system",
                "content": PROMPTS["video_agent"]
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
                tools=[self.search_video_tool],
                stream=False,
                think=True,
                options={'temperature': self.temperature}
            )

            assistant_message = response["message"]
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls")

            if not tool_calls:
                return AgentResponse(
                    answer=assistant_message["content"],
                    used_tools=used_tools,
                    citations=citations,
                    confidence="medium"
                )
            
            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arg = tool_call["function"]["arguments"]

                if name in available_functions:
                    used_tools.append(name)
                    search_results = await available_functions[name](**arg)

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

                    messages.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": tool_response_content,
                    })

                    break
                else:
                    messages.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": f"Error: tool '{name}' not found."
                    })
                    break
        


