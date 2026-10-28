"""
RAG Study Agent - Document Loading, Chunking and Retrieval Module
Beginner-friendly implementation using only built-in Python functionality.
"""

import math
import os
import re
from collections import Counter

# Common words that don't help find relevant chunks
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "you",
}


def load_and_chunk_document(filepath, chunk_size=500):
    """
    Load a text document and split it into chunks of approximately chunk_size characters.

    Args:
        filepath (str): Path to the text file to load (UTF-8 encoded).
        chunk_size (int): Maximum number of characters per chunk (default: 500).

    Returns:
        list[str]: List of text chunks, each being a string.

    Steps:
        1. Load the document from the file using UTF-8 encoding.
        2. Split the document into chunks, trying not to split words in the middle.
        3. Filter out empty chunks and return the list of valid chunks.
    """

    # Step 1: Load the document from the file
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Step 2 & 3: Split into chunks and filter empty ones
    chunks = []
    start = 0

    while start < len(text):
        # Calculate the end position for this chunk
        end = start + chunk_size

        # If we're near the end of the document, take everything remaining
        if end >= len(text):
            chunk = text[start:]
            start = len(text)
        else:
            chunk = text[start:end]

            # Try to find the last whitespace before chunk_size to avoid splitting words
            # Work backwards from the end of the chunk
            last_space = chunk.rfind(' ')
            if last_space != -1 and last_space > chunk_size * 0.8:
                # Found a good place to split (near the end but not in the middle of a word)
                chunk = text[start:start + last_space + 1]
                start = start + last_space + 1
            else:
                # No good split point found, just use the chunk_size boundary
                chunk = text[start:end]
                start = end

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

    # Step 4: Return the list of chunks
    return chunks


def load_documents(folder, chunk_size=500):
    """
    Load and chunk every .txt file in a folder.

    Returns:
        list[dict]: One dict per chunk: {"source": filename, "text": chunk_text}.
    """
    all_chunks = []
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        for text in load_and_chunk_document(path, chunk_size=chunk_size):
            all_chunks.append({"source": filename, "text": text})
    return all_chunks


def tokenize(text):
    """Lowercase the text, split it into words, and drop stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def search(chunks, query, top_k=4):
    """
    Find the chunks most relevant to the query using BM25 keyword scoring.

    BM25 rewards chunks that contain the query words often, gives more weight
    to rare words, and avoids favouring chunks just because they are long.

    Returns:
        list[dict]: Up to top_k chunks (with an added "score"), best first.
    """
    query_words = tokenize(query)
    if not chunks or not query_words:
        return []

    chunk_words = [tokenize(c["text"]) for c in chunks]
    avg_len = sum(len(w) for w in chunk_words) / len(chunks) or 1

    # How many chunks contain each word (used to weight rare words higher)
    doc_freq = Counter()
    for words in chunk_words:
        doc_freq.update(set(words))

    k1, b = 1.5, 0.75
    results = []
    for chunk, words in zip(chunks, chunk_words):
        counts = Counter(words)
        score = 0.0
        for word in query_words:
            if word not in counts:
                continue
            idf = math.log(1 + (len(chunks) - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5))
            tf = counts[word]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(words) / avg_len))
        if score > 0:
            results.append({**chunk, "score": round(score, 3)})

    results.sort(key=lambda c: c["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test the function with the sample document
    filepath = "documents/ai_notes.txt"
    
    print("=" * 60)
    print("Testing load_and_chunk_document function")
    print("=" * 60)
    print()

    # Load and chunk the document
    chunks = load_and_chunk_document(filepath, chunk_size=500)

    # Print results
    print(f"Total number of chunks: {len(chunks)}\n")
    print("-" * 60)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
        print("-" * 60)
