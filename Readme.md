# LocalChatBot

LocalChatBot is a local AI chatbot application built with Chainlit and Ollama. It supports multi-turn conversation, local model selection, document upload, image input, and a VideoRAG workflow for answering questions based on video content.

The project is designed to run locally, using Ollama models for chat, vision, and embeddings. It also includes a video indexing pipeline that splits videos into chunks, transcribes audio, analyzes visual frames, builds a FAISS vector index, and retrieves relevant video segments during chat.

## Features

- Local chatbot UI powered by Chainlit
- Ollama-based local LLM inference
- Model selection from the Chainlit interface
- Optional streaming responses
- Document reading support for PDF, DOCX, and TXT files
- Image input support for vision-capable Ollama models
- Chat history persistence with PostgreSQL
- VideoRAG agent for video-based question answering
- FAISS vector search over video transcripts and visual descriptions

## Project Structure

```txt
LocalChatBot/
├── app.py                         # Main Chainlit app
├── llama_server.py                # Alternative server-compatible chat entry
├── agent/
│   ├── agent_factory.py           # Creates and registers agents
│   ├── agent_registry.py          # Agent routing by chat profile
│   ├── base_agent.py              # Shared agent interface
│   ├── master_agent.py            # Main controller agent
│   ├── video_agent.py             # VideoRAG agent
│   └── agent_prompt/              # Prompt templates
├── tools/
│   └── chainlit_helper.py         # Helper functions for Chainlit responses
├── video_rag/
│   ├── video_splitter.py          # Splits videos into chunks
│   ├── transcriber.py             # Transcribes video chunks
│   ├── visual_analyser.py         # Extracts and analyzes video frames
│   ├── manifest_to_vector.py      # Builds FAISS vector index
│   ├── search_faiss_index.py      # Searches indexed video chunks
│   └── video_index_service.py     # Video index service wrapper
├── data/                          # Local video/index data
└── .chainlit/                     # Chainlit configuration
```

## Requirements

This project requires:

- Python 3.12 recommended
- Ollama installed and available from the command line
- PostgreSQL installed and running
- FFmpeg and FFprobe installed and available on PATH
- At least one Ollama chat model
- One Ollama embedding model, currently bge-m3:latest
- One Ollama vision model, currently qwen3-vl:4b

```txt
chainlit==2.11.1
ollama==0.6.2
requests==2.34.2
PyMuPDF==1.27.2.3
python-docx==1.2.0
pydantic==2.13.4
SQLAlchemy==2.0.49
asyncpg==0.31.0
python-dotenv==1.2.2
numpy==2.4.6
faiss-cpu==1.13.2
faster-whisper==1.2.1
```

## Setup

1. Create a virtual environment
2. Install dependencies
3. Install Ollama
4. Install FFmpeg
5. Set up PostgreSQL

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
go to https://ollama.com/
ollama pull qwen3-vl:4b
ollama pull bge-m3:latest
ffmpeg -version
ffprobe -version
postgresql+asyncpg://chainlit_user:12345678@localhost:5432/chainlit_db
chainlit run app.py
```

Chat Profiles
-------------

The app currently provides two chat profiles:

### Master

The default controller agent. It handles normal chat and can decide when video-related tools are needed.

### VideoRAG

A direct video retrieval agent. Use this profile when you want to ask questions about indexed video content.

Document and Image Upload
-------------------------

The chatbot supports uploaded files during conversation.

Supported document formats:

*   PDF
    
*   DOCX
    
*   TXT
    

Supported image formats:

*   PNG
    
*   JPG / JPEG
    
*   WEBP
    
*   GIF
    
*   BMP
    

Document contents are extracted and added to the conversation context. Images are passed to vision-capable models when supported.

Current VideoRAG Workflow
-------------------------

The video processing pipeline is currently manual. The main steps are:

### 1\. Add videos

Place source videos in:

data/videos/

Supported video formats include:

.mp4, .mov, .mkv, .avi, .webm, .m4v

### 2\. Split videos into chunks

python video\_rag/video\_splitter.py

This creates video chunks and writes:

data/video\_index/chunks\_manifest.jsonl

### 3\. Transcribe video chunks

python video\_rag/transcriber.py

This generates transcript data and writes:

data/video\_index/transcripts\_manifest.jsonl

### 4\. Analyze visual frames

python video\_rag/visual\_analyser.py

This extracts frames from video chunks, sends them to the vision model, and writes:

data/video\_index/visual\_manifest.jsonl

### 5\. Build the FAISS index

python video\_rag/manifest\_to\_vector.py

This creates:

data/video\_index/faiss.indexdata/video\_index/faiss\_metadata.json

After this step, the VideoRAG agent can retrieve relevant video chunks during chat.

Planned Improvements
--------------------

The next development goal is to make the video backend more automated.

Currently, users need to manually run each video processing script. The planned workflow is:

1.  User uploads a video through the Chainlit interface.
    
2.  The backend saves the uploaded video.
    
3.  The system automatically splits the video into chunks.
    
4.  The system automatically transcribes each chunk.
    
5.  The system automatically extracts and analyzes representative frames.
    
6.  The system automatically builds or updates the FAISS index.
    
7.  The user can immediately ask questions about the uploaded video.
    

This will make the VideoRAG workflow much easier to use and reduce the amount of manual setup required.

Notes
-----

*   This project is designed for local development and experimentation.
    
*   Ollama model names may need to be changed depending on what models are installed locally.
    
*   Video processing can take a long time depending on video length and hardware.
    
*   The current implementation uses CPU-based transcription by default.
    
*   PostgreSQL is required for Chainlit chat persistence.

