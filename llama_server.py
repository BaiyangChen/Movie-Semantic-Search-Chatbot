import base64
import json
import mimetypes
import os
import requests
import asyncio
import fitz
import time
import chainlit as cl


from chainlit.input_widget import Select, Switch, Slider
from dotenv import load_dotenv
from typing import Dict, Optional
from docx import Document
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from operator import itemgetter


os.environ['CHAINLIT_AUTH_SECRET'] = "HL8lZ_~NsbXKTvtE:1GuO2D,~pnE*Fq-.s9b^w,ok:2MUkXXMdr2PPkqhMe4UZNR"
os.environ['DATABASE_URL'] = "postgresql+asyncpg://chainlit_user:12345678@localhost:5432/chainlit_db"

LLAMA_SERVER_CHAT_URL = os.getenv(
    "LLAMA_SERVER_CHAT_URL",
    "http://localhost:11434/v1/chat/completions",
)
models = ['Qwen3.6-27B-Q6_K.gguf']

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


def image_to_content_part(image):
    mime_type = getattr(image, 'mime', None) or mimetypes.guess_type(image.path)[0] or 'image/png'
    with open(image.path, 'rb') as file:
        encoded = base64.b64encode(file.read()).decode('utf-8')
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def build_user_content(text, image_files):
    if not image_files:
        return text

    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(image_to_content_part(image) for image in image_files if image.path)
    return content


def split_thinking_from_content(content):
    thinking_parts = []
    answer_parts = []
    remaining = content or ''

    while remaining:
        start = remaining.find('<think>')
        if start == -1:
            answer_parts.append(remaining)
            break

        answer_parts.append(remaining[:start])
        remaining = remaining[start + len('<think>'):]
        end = remaining.find('</think>')
        if end == -1:
            thinking_parts.append(remaining)
            break

        thinking_parts.append(remaining[:end])
        remaining = remaining[end + len('</think>'):]

    return ''.join(thinking_parts).strip(), ''.join(answer_parts).strip()


def build_llama_server_payload(settings, chat_history):
    return {
        "model": settings['Models'],
        "messages": chat_history,
        "stream": settings['Streaming'],
        "temperature": settings['Temperature'],
    }

@cl.password_auth_callback
def auth_callback(username:str, password:str):
    return cl.User(identifier=username)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set('chat_history', [])

    settings = await cl.ChatSettings(
        [
            Select(
                id="Models",
                label="Model",
                values=models,
                initial_value=models[0]
            ),
            Switch(id="Think", label="Show thinking when llama-server returns it", initial=False),
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
    cl.user_session.set('settings', settings)

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=os.getenv('DATABASE_URL'))

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
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

    user_message = {'role': 'user', 'content': build_user_content(message.content, image_files)}
    chat_history.append(user_message)

    assistant_response = ''
    start = time.time()
    final_answer = cl.Message(content="")
    thinking_step = None
    thinking_seen = False
    in_thinking_block = False
    content_buffer = ''

    async def stream_thinking(text):
        nonlocal thinking_step, thinking_seen
        if not text:
            return
        thinking_seen = True
        if settings['Think']:
            if thinking_step is None:
                thinking_step = cl.Step(name='Thinking', type='llm')
                await thinking_step.send()
            await thinking_step.stream_token(text)

    async def stream_answer(text):
        nonlocal assistant_response
        if text:
            assistant_response += text
            await final_answer.stream_token(text)

    async def process_content_delta(delta):
        nonlocal content_buffer, in_thinking_block
        content_buffer += delta or ''

        while content_buffer:
            if in_thinking_block:
                end = content_buffer.find('</think>')
                if end == -1:
                    await stream_thinking(content_buffer)
                    content_buffer = ''
                    return
                await stream_thinking(content_buffer[:end])
                content_buffer = content_buffer[end + len('</think>'):]
                in_thinking_block = False
                continue

            start_tag = content_buffer.find('<think>')
            if start_tag == -1:
                keep = len('<think>') - 1
                if len(content_buffer) <= keep:
                    return
                await stream_answer(content_buffer[:-keep])
                content_buffer = content_buffer[-keep:]
                return

            await stream_answer(content_buffer[:start_tag])
            content_buffer = content_buffer[start_tag + len('<think>'):]
            in_thinking_block = True

    async def flush_content_buffer():
        nonlocal content_buffer, in_thinking_block
        if in_thinking_block:
            await stream_thinking(content_buffer)
        else:
            await stream_answer(content_buffer)
        content_buffer = ''
        in_thinking_block = False

    payload = build_llama_server_payload(settings, chat_history)

    if settings['Streaming']:
        with requests.post(LLAMA_SERVER_CHAT_URL, json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data: '):
                    continue

                data = line.removeprefix('data: ').strip()
                if data == '[DONE]':
                    break

                chunk = json.loads(data)
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                await stream_thinking(delta.get('reasoning_content', ''))
                await process_content_delta(delta.get('content', ''))

        await flush_content_buffer()
    else:
        response = requests.post(LLAMA_SERVER_CHAT_URL, json=payload, timeout=600)
        response.raise_for_status()
        message_data = response.json().get('choices', [{}])[0].get('message', {})

        reasoning = message_data.get('reasoning_content', '')
        content = message_data.get('content', '')
        tag_thinking, answer = split_thinking_from_content(content)
        await stream_thinking(reasoning or tag_thinking)
        await stream_answer(answer if tag_thinking else content)

    if thinking_step and thinking_seen:
        thought_duration = round(time.time()-start)
        thinking_step.name = f"Thought for {thought_duration}s"
        await thinking_step.update()

    await final_answer.send()

    chat_history.append({'role':'assistant', 'content': assistant_response})
