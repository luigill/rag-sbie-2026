import json
import os
import re

import chromadb
import ollama
import streamlit as st
import yaml

# ─── Configuração ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
GENERATION_MODEL = "deepseek-r1:8b"  # ollama pull llama3  (ou gemma3, mistral, etc.)


# ─── Markdown ─────────────────────────────────────────────────────────────────


def extract_text_from_md(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


# ─── Chunking ────────────────────────────────────────────────────────────────


def split_text_into_chunks(
    text: str,
    max_length: int = 400,
    sentence_max_length: int = 400,
) -> list[str]:
    """Divide o texto em chunks menores respeitando o tamanho máximo."""
    sentence_pattern = re.compile(r"([^.!?]+[.!?])")
    sentences = sentence_pattern.findall(text)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    def flush():
        if current_chunk:
            chunks.append("".join(current_chunk).strip())

    for sentence in sentences:
        if len(sentence) > sentence_max_length:
            parts = [
                sentence[i : i + sentence_max_length]
                for i in range(0, len(sentence), sentence_max_length)
            ]
            for part in parts:
                if current_length + len(part) > max_length:
                    flush()
                    current_chunk, current_length = [], 0
                current_chunk.append(part)
                current_length += len(part)
        else:
            if current_length + len(sentence) > max_length:
                flush()
                current_chunk, current_length = [], 0
            current_chunk.append(sentence)
            current_length += len(sentence)

    flush()
    return chunks


# ─── Embeddings (Ollama) ──────────────────────────────────────────────────────


def get_embedding(text: str) -> list[float]:
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response["embeddings"]


# ─── ChromaDB ────────────────────────────────────────────────────────────────


def query_chromadb(collection, embedding: list[float], n_results: int = 5) -> tuple:
    """Realiza consulta por similaridade no ChromaDB."""
    n_results = min(n_results, collection.count())
    if n_results == 0:
        return [], [], []
    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    return results["documents"], results["metadatas"], results["distances"]


# ─── Processamento de PDFs ────────────────────────────────────────────────────


def process_mds_in_folder(folder_path: str, collection, on_progress=None) -> str:
    """Processa markdowns em uma pasta, gera embeddings e persiste no ChromaDB."""
    all_chunks: dict[str, list[str]] = {}
    processed: list[str] = []

    md_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
    pending = []
    for md_file in md_files:
        doc_name = os.path.splitext(md_file)[0]
        existing = collection.get(ids=[f"{doc_name}_doc_0"])
        if not existing["ids"]:
            pending.append(md_file)

    if not pending:
        if on_progress:
            on_progress(1.0, "Todos os markdowns já foram processados.")
        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)
        return "Todos os markdowns já foram processados."

    total = len(pending)
    # 3 steps per file: chunk, embed, store
    total_steps = total * 3

    for file_idx, md_file in enumerate(pending):
        doc_name = os.path.splitext(md_file)[0]
        md_path = os.path.join(folder_path, md_file)
        base_step = file_idx * 3

        if on_progress:
            on_progress(base_step / total_steps, f"[{file_idx+1}/{total}] {doc_name} — dividindo em chunks...")
        text = extract_text_from_md(md_path)
        chunks = split_text_into_chunks(text)

        if on_progress:
            on_progress((base_step + 1) / total_steps, f"[{file_idx+1}/{total}] {doc_name} — gerando embeddings ({len(chunks)} chunks)...")
        embeddings = get_embeddings(chunks)

        if on_progress:
            on_progress((base_step + 2) / total_steps, f"[{file_idx+1}/{total}] {doc_name} — salvando no ChromaDB...")
        collection.add(
            documents=chunks,
            metadatas=[{"chunk_id": i, "doc_name": doc_name} for i in range(len(chunks))],
            ids=[f"{doc_name}_doc_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
        )

        all_chunks[doc_name] = chunks
        processed.append(doc_name)

    if on_progress:
        on_progress(1.0, f"Concluído! {len(processed)} arquivo(s) processado(s).")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    return f"Processados: {', '.join(processed)}"


# ─── Geração de resposta (Ollama) ─────────────────────────────────────────────


def format_prompt(prompt_template: str, query: str, chunks: str) -> str:
    return prompt_template.format(query=query, chunks=chunks)


def process_query(collection, query: str) -> tuple[str, list[dict]]:
    """Recupera contexto do ChromaDB e gera resposta via Ollama.

    Retorna (resposta, lista de chunks com texto, fonte e distância).
    """
    with open("prompt_template.yml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    system_prompt: str = prompts["System_Prompt"]
    prompt_template: str = prompts["prompt_instructions"]

    embedding = get_embedding(query)
    documents, metadatas, distances = query_chromadb(collection, embedding)

    inner_docs = (
        documents[0] if documents and isinstance(documents[0], list) else documents
    )
    inner_meta = (
        metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
    )
    inner_dist = (
        distances[0] if distances and isinstance(distances[0], list) else distances
    )

    chunks_info = [
        {"text": str(doc), "meta": meta, "distance": dist}
        for doc, meta, dist in zip(inner_docs, inner_meta, inner_dist)
    ]

    chunks_text = "\n\n".join(c["text"] for c in chunks_info)[:4000]
    user_prompt = format_prompt(prompt_template, query, chunks_text)

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"num_ctx": 16000},
    )
    return response["message"]["content"], chunks_info


# ─── Interface Streamlit ──────────────────────────────────────────────────────


def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("pdf_embeddings")

    st.set_page_config(page_title="Pelot.ai - Descubra Pelotas", page_icon="🧁")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
            font-size: 22px;
            font-weight: 700;
            color: #091747;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton > button:hover { background-color: #45a049; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Pelot.ai 👸🧁🐜")
    st.caption(
        f"Embeddings: `{EMBEDDING_MODEL}` · Geração: `{GENERATION_MODEL}` · 100% offline via Ollama"
    )
    st.write("Faça perguntas sobre a história da cidade de Pelotas!")

    question = st.text_input(
        "### Insira aqui sua pergunta:", placeholder="Digite sua pergunta aqui..."
    )

    if st.button("Enviar"):
        if question.strip():
            with st.spinner("Consultando o modelo local..."):
                answer, chunks_info = process_query(collection, question)
            st.session_state["answer"] = answer
            st.session_state["chunks_info"] = chunks_info
        else:
            st.warning("Faça uma pergunta!")

    tab_resposta, tab_chunks = st.tabs(["Resposta", "Chunks recuperados"])

    with tab_resposta:
        st.write(st.session_state.get("answer", "Sua resposta aparecerá aqui...."))

    with tab_chunks:
        chunks_info = st.session_state.get("chunks_info", [])
        if not chunks_info:
            st.write("Nenhum chunk recuperado ainda.")
        else:
            for i, chunk in enumerate(chunks_info, 1):
                source = chunk["meta"].get("doc_name", "?")
                chunk_id = chunk["meta"].get("chunk_id", "?")
                dist = chunk["distance"]
                with st.expander(
                    f"#{i} · {source} · chunk {chunk_id} · dist {dist:.4f}"
                ):
                    st.write(chunk["text"])

    if st.button("Processar Documentos"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(pct: float, message: str):
            progress_bar.progress(pct)
            status_text.text(message)

        status = process_mds_in_folder("data_md", collection, on_progress=on_progress)
        progress_bar.empty()
        status_text.empty()
        st.markdown("### Status:")
        st.write(status)


if __name__ == "__main__":
    main()
