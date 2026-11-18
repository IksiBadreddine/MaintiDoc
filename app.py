"""MaintiDoc — interface Streamlit, importation et indexation multimodale des PDF.

Lancement : uv run streamlit run app.py
Un PDF importé est indexé une seule fois puis interrogé dans un processus dédié.
Le manuel PACIFIC et ses collections existantes restent intacts.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_manager import (  # noqa: E402
    list_ready_documents, register_upload, index_document, query_agent,
    is_ready, MAX_PDF_SIZE,
)

MAX_PHOTO_BYTES = 12 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(
    page_title="MaintiDoc — Assistant de maintenance",
    page_icon="🔧",
    layout="wide",
)


def safe_image_path(raw_path: str, doc: dict) -> Path | None:
    """N'affiche que les images qui appartiennent au document actif."""
    if not isinstance(raw_path, str) or not raw_path:
        return None
    try:
        image_path = Path(raw_path).expanduser().resolve()
        allowed = (doc["document_dir"] / "images").resolve()
        if image_path.is_relative_to(allowed) and image_path.is_file():
            return image_path
    except (OSError, ValueError):
        pass
    return None


def summarize_trace(result: dict) -> dict:
    """Trace structurée indépendante des citations éventuelles du LLM."""
    trace = result.get("tool_trace") or []
    refs = {"tools": [], "images": [], "passages": [], "pages": []}
    seen_images = set()
    seen_passages = set()
    for entry in trace:
        refs["tools"].append(entry.get("tool", "?"))
        if entry.get("tool") == "search_manual" and entry.get("context"):
            excerpt = str(entry["context"])[:1100]
            if excerpt not in seen_passages:
                refs["passages"].append(excerpt)
                seen_passages.add(excerpt)
        if entry.get("tool") == "read_manual_page" and entry.get("context"):
            excerpt = str(entry["context"])[:1100]
            if excerpt not in seen_passages:
                refs["passages"].append(excerpt)
                seen_passages.add(excerpt)
        for img in entry.get("images", []):
            path = img.get("image_path", "")
            if not path or path in seen_images:
                continue
            seen_images.add(path)
            refs["images"].append({
                "page": img.get("page"),
                "number": img.get("image_number"),
                "path": path,
                "distance": img.get("distance_clip"),
            })
    pages = result.get("manual_pages") or []
    refs["pages"] = sorted(
        {str(p) for p in pages if p is not None},
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )
    refs["images"] = refs["images"][:6]
    refs["passages"] = refs["passages"][:4]
    return refs


def show_sources(refs: dict, doc: dict) -> None:
    with st.expander("📚 Sources et outils utilisés"):
        st.caption("Document actif : " + doc["name"])
        pages = refs.get("pages") or []
        st.write("**Pages retrouvées :** " + (", ".join(pages) if pages else "aucune"))
        tools = refs.get("tools") or []
        st.write("**Outils exécutés :** " + (" → ".join(tools) if tools else "aucun"))
        st.caption(
            "Les pages et illustrations proviennent des outils ; chaque affirmation "
            "doit encore être confrontée au passage cité."
        )
        images = refs.get("images") or []
        if images:
            st.write("**Illustrations retrouvées dans le document actif :**")
            columns = st.columns(min(3, len(images)))
            for i, item in enumerate(images):
                path = safe_image_path(item.get("path", ""), doc)
                if path is None:
                    continue
                caption = f"Page {item.get('page', '?')} · Image {item.get('number', '?')}"
                distance = item.get("distance")
                if isinstance(distance, (float, int)):
                    caption += f" · distance CLIP {distance:.4f}"
                with columns[i % len(columns)]:
                    st.image(str(path), caption=caption, use_container_width=True)
        passages = refs.get("passages") or []
        if passages:
            st.write("**Extraits réellement consultés :**")
            for i, passage in enumerate(passages, 1):
                st.code(passage, language=None)


def display_message(message: dict, doc: dict) -> None:
    with st.chat_message(message.get("role", "assistant")):
        st.markdown(message.get("text", ""))
        if message.get("role") == "user" and message.get("photo_bytes"):
            st.image(message["photo_bytes"], caption="Photo transmise", width=260)
        if message.get("role") == "assistant" and message.get("sources"):
            show_sources(message["sources"], doc)


def validate_photo(uploaded) -> tuple[bytes, str]:
    raw = uploaded.getvalue()
    if not raw or len(raw) > MAX_PHOTO_BYTES:
        raise ValueError("Photo vide ou de plus de 12 Mo.")
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError("Le format de la photo doit être JPEG, PNG ou WebP.")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError("Ce fichier n'est pas une photo valide.") from exc
    return raw, suffix


def invoke_agent(doc: dict, question: str, photo_bytes: bytes | None,
                 suffix: str, text_k: int, image_k: int) -> dict:
    photo_path = None
    try:
        if photo_bytes is not None:
            with tempfile.NamedTemporaryFile(prefix="maintidoc_photo_", suffix=suffix, delete=False) as temp:
                temp.write(photo_bytes)
                photo_path = Path(temp.name)
        return query_agent(
            doc, question, str(photo_path) if photo_path else None,
            top_k=text_k, image_k=image_k,
        )
    finally:
        if photo_path is not None:
            photo_path.unlink(missing_ok=True)


st.title("🔧 MaintiDoc")
st.caption("Assistant de maintenance — RAG agentique multimodal")

if "active_document" not in st.session_state:
    st.session_state.active_document = "pacific"
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

available = list_ready_documents()
by_id = {doc["id"]: doc for doc in available}
if not available:
    st.error("Aucun manuel indexé disponible. Commence par 'uv run python main.py' pour PACIFIC.")
    st.stop()
if st.session_state.active_document not in by_id:
    st.session_state.active_document = available[0]["id"]

with st.sidebar:
    st.header("Documents")
    ids = list(by_id)
    current = st.session_state.active_document
    choice = st.selectbox(
        "Document actif",
        options=ids,
        index=ids.index(current),
        format_func=lambda ident: ("PACIFIC — " if ident == "pacific" else "") + by_id[ident]["name"],
    )
    if choice != current:
        st.session_state.active_document = choice
        st.rerun()
    selected = by_id[choice]
    st.success("Index actif : " + selected["name"])
    st.caption("Les recherches portent exclusivement sur ce document.")

    st.divider()
    st.subheader("Ajouter un PDF technique")
    st.caption("PDF avec textes, tableaux et images extractibles. Limite : 25 Mo.")
    pdf_upload = st.file_uploader("Importer un PDF", type=["pdf"], key="uploaded_pdf")
    if pdf_upload is not None:
        raw = pdf_upload.getvalue()
        if len(raw) > MAX_PDF_SIZE:
            st.error("Fichier trop volumineux : 25 Mo maximum.")
        else:
            try:
                import pymupdf
                with pymupdf.open(stream=raw, filetype="pdf") as pdf:
                    st.caption(f"{pdf_upload.name} — {len(pdf)} pages")
            except Exception:
                st.warning("Impossible de prévisualiser ce PDF.")
            if st.button("Indexer et utiliser ce PDF", use_container_width=True, type="primary"):
                try:
                    doc = register_upload(pdf_upload.name, raw)
                    with st.spinner("Extraction, Qwen-VL, BGE-M3, CLIP et vérifications en cours..."):
                        index_document(doc)
                    st.success("Document indexé avec succès.")
                    st.session_state.active_document = doc["id"]
                    st.rerun()
                except Exception as exc:
                    st.error("Indexation impossible : " + str(exc))
                    st.caption("Le manuel PACIFIC n'a pas été modifié.")

    st.divider()
    st.subheader("Recherche")
    text_k = st.slider("Passages BGE-M3", 2, 10, 5)
    image_k = st.slider("Images CLIP", 1, 6, 3)
    if st.button("Effacer la conversation du document actif", use_container_width=True):
        st.session_state.conversations[choice] = []
        st.rerun()

active = by_id[st.session_state.active_document]
if not is_ready(active):
    st.error("L'index du manuel actif n'est plus disponible.")
    st.stop()

messages = st.session_state.conversations.setdefault(active["id"], [])
if not messages:
    st.info("Pose une question sur le manuel sélectionné ou joins une photographie.")
for message in messages:
    display_message(message, active)

submission = st.chat_input(
    f"Question sur {active['name']}...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
)
if submission is not None:
    question = (submission.text or "").strip()
    files = list(submission.files or [])
    if not question and files:
        question = "Que montre cette photographie ? Réponds uniquement avec les informations du manuel actif."
    if not question:
        st.warning("Écris une question ou joins une photo.")
        st.stop()
    photo_bytes, photo_suffix = None, ""
    if files:
        try:
            photo_bytes, photo_suffix = validate_photo(files[0])
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    human = {"role": "user", "text": question, "photo_bytes": photo_bytes}
    messages.append(human)
    display_message(human, active)
    with st.chat_message("assistant"):
        try:
            with st.spinner("L'agent interroge le manuel actif..."):
                result = invoke_agent(active, question, photo_bytes, photo_suffix, text_k, image_k)
            answer = str(result.get("answer") or "Aucune réponse obtenue.")
            refs = summarize_trace(result)
            st.markdown(answer)
            show_sources(refs, active)
            messages.append({"role": "assistant", "text": answer, "sources": refs})
        except Exception as exc:
            st.error("Impossible de traiter la demande : " + str(exc))
            st.caption("Vérifie le serveur Ollama et les modèles qwen3:8b / qwen2.5vl:3b.")
    # Historique d'affichage indépendant par manuel (pas encore mémoire du LLM).
    st.session_state.conversations[active["id"]] = messages[-24:]
