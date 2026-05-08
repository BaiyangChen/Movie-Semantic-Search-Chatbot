from operator import itemgetter
import os
import ollama
import subprocess
import threading
import requests
import asyncio
import fitz
import time
import chainlit as cl
import socket


from chainlit.input_widget import Select, Switch, Slider
from dotenv import load_dotenv
from typing import Dict, Optional
from docx import Document
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from video_rag.search_faiss_index import (
    search_video_chunks,
    format_video_chunk_for_prompt,
)

os.environ['CHAINLIT_AUTH_SECRET'] = "HL8lZ_~NsbXKTvtE:1GuO2D,~pnE*Fq-.s9b^w,ok:2MUkXXMdr2PPkqhMe4UZNR"
os.environ['DATABASE_URL'] = "postgresql+asyncpg://chainlit_user:12345678@localhost:5432/chainlit_db"

models = [
    'fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4',
    'fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b',
]

thinking_models = {
    'fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4',
}

def _ollama():
    os.environ["OLLAMA_HOST"] = '0.0.0.0:11434'
    os.environ["OLLAMA_ORIGIN"] = '*'
    subprocess.Popen(['ollama', 'serve'])

def is_ollama_running():
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1):
            return True
    except OSError:
        return False

def start_ollama():
    if is_ollama_running():
        return
    thread = threading.Thread(target=_ollama)
    thread.daemon = True
    thread.start()

def read_document(documents):
    text = ''
    for document in documents:
        path = document.path.lower()
        if path.endswith('pdf'):
            doc = fitz.open(document.path)
            text += '\n\nFile:' + document.name + '\n'
            for page in doc:
                text += page.get_text()
        elif path.endswith('docx'):
            doc = Document(document.path)
            text += '\n\nFile:' + document.name + '\n' + '\n'.join([p.text for p in doc.paragraphs])
        elif path.endswith('txt'):
            with open(document.path, 'r') as f:
                text += '\n\nFile:' + document.name + '\n' + f.read()
    return text

def is_image_file(element):
    path = (element.path or '').lower()
    mime = (getattr(element, 'mime', '') or '').lower()
    return path.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp')) or mime.startswith('image/')


def is_document_file(element):
    path = (element.path or '').lower()
    return path.endswith(('.pdf', '.docx', '.txt'))

async def video_agent(user_query: str, settings: dict) -> str:
    # Implementation for the video agent
    async with cl.Step(name = "Video Agent: Searching for relavant video content", type="tool") as step:
        await step.send()
        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(None, search_video_chunks, user_query, 5)
        
        video_context = format_video_chunk_for_prompt(search_results)
        if not video_context:
            return "I couldn't find any relevant video content based on your query."
        
        step.output = video_context
        await step.update()
        
        messages = [
            {
                "role": "system",
                "content": """
                    You are a video analysis agent.
                    Answer the user using only the retrieved video chunks.
                    If the chunks do not contain enough evidence, say that clearly.
                    Mention relevant timestamps when useful.
                """,
            },
            {
                "role": "user",
                "content": f"""
                    User question:
                    {user_query}

                    Retrieved video chunks:
                    {video_context}
                """,
            },
        ]

    response = ollama.chat(
        model=settings["Models"],
        messages=messages,
        stream=False,
        think=False,
        options={"temperature": settings["Temperature"]},
    )

    return response["message"]["content"]

async def master_route(user_query: str, settings: Dict) -> str:
    route_messages = [
        {
            "role": "system",
            "content": """
                You are a routing agent.
                Decide whether the user's request needs video retrieval.

                Return only one word:
                video - if the user asks about video content, transcript, timestamps, scenes, visual notes, or anything likely stored in the video index.
                direct - for normal conversation, coding, general knowledge, document/image uploads, or anything not requiring video search.
            """,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    response = ollama.chat(
        model=settings["Models"],
        messages=route_messages,
        stream=False,
        think=False,
        options={"temperature": 0},
    )

    decision = response["message"]["content"].strip().lower()

    if "video" in decision:
        return "video"

    return "direct"



@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Master",
            description="Main controller agent for general conversations and automatically decides whether to use the video agent",
            icon="💬"
        ),
        cl.ChatProfile(
            name="VideoRAG",
            description="Directly answers questions using the video retrieval agent.",
            icon="🎥"
        )
    ]

@cl.password_auth_callback
def auth_callback(username:str, password:str):
    return cl.User(identifier=username)

@cl.on_chat_start
async def on_chat_start():
    start_ollama()
    cl.user_session.set('chat_history', [])
    profile = cl.user_session.get("chat_profile")

    settings = await cl.ChatSettings(
        [
            Select(
                id="Models",
                label="Model",
                values=models,
                initial_value='fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4'
            ),
            Switch(id="Think", label="Enable thinking for Qwen3 models", initial=False),
            Switch(id="Streaming", label="Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="SAI_Cfg_Scale",
                label="Stability AI - Cfg_Scale",
                initial=7,
                min=1,
                max=35,
                step=0.1,
                description="Influences how strongly your generation is guided to match your prompt.",
            )
        ]
    ).send()
    cl.user_session.set("active_profile", profile or "Master")
    cl.user_session.set('settings', settings)

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.getenv('DATABASE_URL'))

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    start_ollama()
    cl.user_session.set("chat_history",[])

    for message in thread['steps']:
        if message['type'] == 'user_message':
            cl.user_session.get('chat_history').append({'role':'user', 'content': message['output']})
        elif message['type'] == 'assistant_message':
            cl.user_session.get('chat_history').append({'role':'assistant', 'content': message['output']})

@cl.on_message
async def on_message(message : cl.message):
    chat_history = cl.user_session.get('chat_history', [])
    settings = cl.user_session.get('settings')
    model = settings['Models']
    settings['Think'] = settings['Think'] and model in thinking_models
    
    files = [file for file in message.elements]
    document_files = [file for file in files if is_document_file(file)]
    image_files = [file for file in files if is_image_file(file)]

    if document_files:
        async with cl.Step('Reading Documents') as reading_documents:
            await reading_documents.send()
            loop = asyncio.get_event_loop()
            document_data = await loop.run_in_executor(None, read_document, document_files)
            content = f"""The user has uploaded the following file content:{document_data} You already have access to the file content above. 
                You must answer based on it. Do not say that you cannot access the uploaded file."""

            chat_history.append({'role':'system', 'content': content})
            await reading_documents.remove()

    user_message = {'role': 'user', 'content': message.content}
    if image_files:
        user_message['images'] = [image.path for image in image_files if image.path]

    chat_history.append(user_message)

    stream = ollama.chat(
            model=settings['Models'],
            messages=chat_history,
            stream=settings['Streaming'],
            think=settings['Think'],
            options={ 'temperature': settings['Temperature'] }
        )
    
    thinking = False
    assistant_response = ''
    start = time.time()
    final_answer = cl.Message(content="")

    if settings["Think"]:
        async with cl.Step(name='Thinking', type='llm') as thinking_step:
            for chunk in stream:
                think = chunk.get("message", {}).get("thinking", "")
                if think:
                    thinking = True
                    await thinking_step.stream_token(think)
                elif settings["Think"] and thinking:
                    thinking = False
                    thought_duration = round(time.time()-start)
                    thinking_step.name = f"Thought for {thought_duration}s"
                    await thinking_step.update()
                    break

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            assistant_response += content
            await final_answer.stream_token(content)

    await final_answer.send()

    chat_history.append({'role':'assistant', 'content': assistant_response})
