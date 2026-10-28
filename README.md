# RAG Study Agent

A web app for studying your own notes. You upload `.txt` notes and ask questions in a chat. The app finds the most relevant passages using retrieval-augmented generation (RAG), and Claude streams back an answer that cites them.

## How it works

1. **Chunking** (`rag.py`): each note is split into chunks of about 500 characters, without cutting words in half.
2. **Retrieval** (`rag.py`): BM25 keyword scoring picks the 4 chunks most relevant to your question.
3. **Generation** (`agent.py`): those chunks and your question go to Claude (`claude-opus-5-5`). The answer is streamed back word by word and cites the chunks it used, like [1] or [2].
4. **Web app** (`app.py` + `templates/index.html`): a Flask server with a chat UI. You can upload or remove notes, ask follow-up questions, and see the source passages behind each answer.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your Anthropic API key into .env
```

Get an API key at <https://console.anthropic.com>.

## Run

```bash
python app.py
```

Open <http://127.0.0.1:5000>. To add notes, drop a `.txt` file into the sidebar or copy it into the `documents/` folder.

## Project layout

```
app.py               Flask web server and API routes
agent.py             Claude prompt + streaming answer generation
rag.py               Document loading, chunking and BM25 search
templates/index.html Chat interface
documents/           Your notes (.txt)
```
