#!/usr/bin/env python3
""""
  1) import  -> reads "sourcedocs.txt", chunks, embeds with Ollama, and loads into ChromaDB
  2) search  -> embeds a user query, retrieves top chunks from ChromaDB, and streams a model answer

Prereqs (same as the original project):
  - Chroma server running on http://localhost:8000
  - Ollama running locally with the models in config.ini (defaults provided)
  - libmagic installed (for python-magic)

Usage examples:
  python rag.py import
  python rag.py search "What does the doc say about X?"

"""
import argparse
import configparser
import os
import re
import time
from urllib.parse import unquote, urlparse

import chromadb
import ollama
import requests
import magic
from bs4 import BeautifulSoup
from mattsollamatools import chunk_text_by_sentences

COLLECTION_NAME = "buildragwithpython"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# ---------- Utilities (from original utilities.py) ----------

def get_filename_from_cd(cd: str | None) -> str | None:
    """Get a filename from a Content-Disposition header value, if present."""
    if not cd:
        return None
    try:
        fname = cd.split('filename=')[1]
    except Exception:
        return None
    if fname.lower().startswith(("utf-8''", "utf-8'")):
        fname = fname.split("'")[-1]
    return unquote(fname.strip('"'))

def download_file(url: str) -> str:
    """Download a URL to ./content/ and return the local filepath."""
    os.makedirs('content', exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        filename = get_filename_from_cd(r.headers.get('content-disposition'))
        if not filename:
            # fall back to a URL-based name
            filename = urlparse(url).geturl().replace('https://', '').replace('/', '-')
        filename = os.path.join('content', filename)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return filename

def readtext(path: str) -> str:
    """Read text from a local path or URL. Supports text/plain and text/html.
    (PDF intentionally not supported to match the original approach.)
    """
    p = path.rstrip().replace(' \n', '').replace('%0A', '')

    if re.match(r'^https?://', p):
        filename = download_file(p)
    else:
        filename = os.path.abspath(p)

    filetype = magic.from_file(filename, mime=True)
    print(f"\nEmbedding {filename} as {filetype}")
    text = ""
    if filetype == 'application/pdf':
        print('PDF not supported yet')
    if filetype == 'text/plain':
        with open(filename, 'rb') as f:
            text = f.read().decode('utf-8')
    if filetype == 'text/html':
        with open(filename, 'rb') as f:
            soup = BeautifulSoup(f, 'html.parser')
            text = soup.get_text()

    # clean up temp downloads saved under ./content
    if os.path.exists(filename) and ('content' + os.sep) in filename:
        try:
            os.remove(filename)
        except Exception:
            pass
    return text

def getconfig(config_path: str = 'config.ini') -> dict:
    """Return the [main] section as a dict. Falls back to sensible defaults."""
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)
        if config.has_section('main'):
            return {k: v for k, v in config.items('main')}
    # defaults if config.ini is missing
    return {"embedmodel": "nomic-embed-text", "mainmodel": "gemma:2b"}

# ---------- Core helpers ----------

def connect_chroma() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

def ensure_collection(client: chromadb.HttpClient, delete_if_exists: bool = False):
    if delete_if_exists and any(c.name == COLLECTION_NAME for c in client.list_collections()):
        print('deleting collection')
        client.delete_collection(COLLECTION_NAME)
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# ---------- Commands ----------

def cmd_import(_: argparse.Namespace) -> None:
    cfg = getconfig()
    embedmodel = cfg["embedmodel"]

    client = connect_chroma()
    print(client.list_collections())
    collection = ensure_collection(client, delete_if_exists=True)

    start = time.time()
    with open('sourcedocs.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            text = readtext(line)
            chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=7, overlap=0)
            print(f"with {len(chunks)} chunks")
            for index, chunk in enumerate(chunks):
                # embed chunk and add to the vector DB
                embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
                print('.', end='', flush=True)
                uid = f"{line.strip()}#{index}"
                collection.add([uid], [embed], documents=[chunk], metadatas={"source": line.strip()})
    print(f"\n--- {time.time() - start:.2f} seconds ---")

def cmd_search(args: argparse.Namespace) -> None:
    cfg = getconfig()
    embedmodel = cfg["embedmodel"]
    mainmodel = cfg["mainmodel"]

    client = connect_chroma()
    collection = ensure_collection(client, delete_if_exists=False)

    query = " ".join(args.query)
    queryembed = ollama.embeddings(model=embedmodel, prompt=query)['embedding']
    res = collection.query(query_embeddings=[queryembed], n_results=5)
    relevantdocs = res.get("documents", [[]])[0]
    docs = "\n\n".join(relevantdocs)

    modelquery = f"{query} - Answer that question using the following text as a resource: {docs}"
    stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)

    for chunk in stream:
        if chunk.get("response"):
            print(chunk['response'], end='', flush=True)

# ---------- CLI ----------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG demo (one file) using ChromaDB + Ollama")
    sp = p.add_subparsers(dest='cmd', required=True)

    p_import = sp.add_parser('import', help='Index documents listed in sourcedocs.txt')
    p_import.set_defaults(func=cmd_import)

    p_search = sp.add_parser('search', help='Search the indexed documents and stream an answer')
    p_search.add_argument('query', nargs='+', help='Your question')
    p_search.set_defaults(func=cmd_search)

    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
