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

from agent.agent_factory import create_agent_registry
from chainlit.input_widget import Select, Switch, Slider
from typing import Dict, Optional
from docx import Document
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

os.environ['CHAINLIT_AUTH_SECRET'] = "HL8lZ_~NsbXKTvtE:1GuO2D,~pnE*Fq-.s9b^w,ok:2MUkXXMdr2PPkqhMe4UZNR"
os.environ['DATABASE_URL'] = "postgresql+asyncpg://chainlit_user:12345678@localhost:5432/chainlit_db"

models = [
    'fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4',
    'fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b',
    "qwen3-vl:4b"
]

thinking_models = {
    'fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4',
    "qwen3-vl:4b"
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

@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Master",
            markdown_description="Main controller agent for general conversations and automatically decides whether to use the video agent",
            icon="💬"
        ),
        cl.ChatProfile(
            name="VideoRAG",
            markdown_description="Directly answers questions using the video retrieval agent.",
            icon="🎥"
        )
    ]

@cl.password_auth_callback
def auth_callback(username:str, password:str):
    return cl.User(identifier=username)

@cl.on_chat_start
async def on_chat_start():
    start_ollama()
    profile = cl.user_session.get("chat_profile") or "Master"

    settings = await cl.ChatSettings(
        [
            Select(
                id="Models",
                label="Model",
                values=models,
                initial_value='qwen3-vl:4b'
            ),
            Switch(id="Think", label="Enable thinking for Qwen3 models", initial=False),
            Switch(id="Streaming", label="Stream Tokens", initial=False),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0.7,
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

    registry = create_agent_registry(
        model=settings['Models']
    )

    cl.user_session.set("active_profile", profile)
    cl.user_session.set('settings', settings)
    cl.user_session.set("agent_registry", registry)
    cl.user_session.set('chat_history', [])


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

    setting = cl.user_session.get('settings', {})

    # agent set up
    registry = cl.user_session.get('agent_registry')
    active_profile = cl.user_session.get("active_profile")
    agent = registry.get_agent_by_profile(active_profile)

    # get model config
    can_think = setting.get("Think", False) and agent.model in thinking_models
    is_stream = setting.get("Streaming", False)
    temperature = setting.get("Temperature", 0.7)
    
    # document and image handling
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

    # response set up
    assistant_response = ''
    final_answer = cl.Message(content="")
    
    await final_answer.send()
    
    if is_stream:
        stream_iter = agent.run_with_stream(query=message.content, chat_history=chat_history, can_think=can_think, is_stream=is_stream, temperature=temperature)
        async for stream in stream_iter:
            await final_answer.stream_token(stream)
            assistant_response += stream
    else:
        response = await agent.run_without_stream(query=message.content, chat_history=chat_history, can_think=can_think, is_stream=is_stream, temperature=temperature)
        assistant_response += response.answer
        final_answer.content = assistant_response

    await final_answer.update()
    chat_history.append({'role':'assistant', 'content': assistant_response})
