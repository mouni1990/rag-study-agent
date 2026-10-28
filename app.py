"""
RAG Study Agent - Web App
Run with:  python app.py   then open http://127.0.0.1:5000
"""

import json
import os
import threading

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

load_dotenv()  # reads ANTHROPIC_API_KEY from a .env file if there is one

import agent  # noqa: E402  (imported after load_dotenv so the key is available)
import rag  # noqa: E402

DOCS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
MAX_UPLOAD_MB = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# All chunks from all documents, kept in memory and rebuilt after each upload/delete
_chunks = []
_chunks_lock = threading.Lock()


def reload_chunks():
    global _chunks
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    new_chunks = rag.load_documents(DOCS_FOLDER)
    with _chunks_lock:
        _chunks = new_chunks


def document_list():
    with _chunks_lock:
        counts = {}
        for c in _chunks:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
    return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/documents", methods=["GET"])
def list_documents():
    return jsonify(document_list())


@app.route("/api/documents", methods=["POST"])
def upload_document():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="No file selected."), 400
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".txt"):
        return jsonify(error="Only .txt files are supported."), 400
    data = file.read()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return jsonify(error="The file must be UTF-8 text."), 400
    with open(os.path.join(DOCS_FOLDER, filename), "wb") as f:
        f.write(data)
    reload_chunks()
    return jsonify(document_list())


@app.route("/api/documents/<name>", methods=["DELETE"])
def delete_document(name):
    path = os.path.join(DOCS_FOLDER, secure_filename(name))
    if os.path.isfile(path):
        os.remove(path)
    reload_chunks()
    return jsonify(document_list())


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    history = [
        m for m in body.get("history", [])
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"]
    ]
    if not question:
        return jsonify(error="Please type a question."), 400

    with _chunks_lock:
        excerpts = rag.search(_chunks, question, top_k=4)

    def generate():
        # First line: the excerpts used, so the page can show sources right away
        sources = [{"source": c["source"], "score": c["score"], "text": c["text"]} for c in excerpts]
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"

        if not os.environ.get("ANTHROPIC_API_KEY"):
            yield json.dumps({"type": "text", "text": (
                "**No API key found.** Create a file named `.env` in the project folder containing\n\n"
                "`ANTHROPIC_API_KEY=your-key-here`\n\nthen restart the app."
            )}) + "\n"
            return

        for piece in agent.stream_answer(question, excerpts, history):
            yield json.dumps({"type": "text", "text": piece}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


reload_chunks()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True, threaded=True)
