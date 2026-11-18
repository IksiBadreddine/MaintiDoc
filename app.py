"""MaintiDoc Streamlit : un manuel OU tous les manuels déjà indexés.

L'upload PDF reste inchangé ; le mode global n'extrait ni ne réindexe les PDF.
Lancement : uv run streamlit run app.py
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
    query_all_documents, is_ready, MAX_PDF_SIZE,
)

MAX_PHOTO_BYTES = 12 * 1024 * 1024
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(page_title="MaintiDoc — RAG multimodal", page_icon="🔧", layout="wide")


def safe_image_path(raw: str, document: dict | None) -> Path | None:
    """Affiche uniquement les illustrations appartenant au PDF cité."""
    if not document or not isinstance(raw, str) or not raw:
        return None
    try:
        path = Path(raw).expanduser().resolve()
        root = (document["document_dir"] / "images").resolve()
        if path.is_relative_to(root) and path.is_file():
            return path
    except (OSError, ValueError):
        pass
    return None


def summarize_trace(result: dict, default_document: dict | None) -> dict:
    """Sources issues de la trace, avec identifiant du PDF et page distincts."""
    trace = result.get("tool_trace") or []
    refs = {"tools": [], "pages": [], "images": [], "passages": []}
    page_seen, image_seen, passage_seen = set(), set(), set()

    def add_page(doc_id, doc_name, page):
        if page in (None, "?"):
            return
        key = (doc_id, str(page))
        if key not in page_seen:
            page_seen.add(key)
            refs["pages"].append({"document_id": doc_id, "document": doc_name, "page": page})

    for entry in trace:
        refs["tools"].append(entry.get("tool", "?"))
        for hit in entry.get("sources", []):
            doc_id = hit.get("document_id") or (default_document or {}).get("id")
            name = hit.get("document") or (default_document or {}).get("name", "Document inconnu")
            page = hit.get("page")
            add_page(doc_id, name, page)
            content = hit.get("content")
            if isinstance(content, str) and content.strip():
                key = (doc_id, str(page), content[:100])
                if key not in passage_seen:
                    passage_seen.add(key)
                    refs["passages"].append({"document": name, "page": page,
                                             "text": content[:1000]})
        if entry.get("tool") in {"search_manual", "read_manual_page", "search_all_manuals"}:
            content = entry.get("context")
            if isinstance(content, str) and content.strip() and len(refs["passages"]) < 6:
                doc_id = entry.get("document_id") or (default_document or {}).get("id")
                name = entry.get("document") or (default_document or {}).get("name", "Recherche documentaire")
                page = entry.get("page")
                key = (doc_id, str(page), content[:100])
                if key not in passage_seen:
                    passage_seen.add(key)
                    refs["passages"].append({"document": name, "page": page,
                                             "text": content[:1000]})
                if page is not None:
                    add_page(doc_id, name, page)
        for hit in entry.get("images", []):
            doc_id = hit.get("document_id") or (default_document or {}).get("id")
            name = hit.get("document") or (default_document or {}).get("name", "Document inconnu")
            page = hit.get("page")
            add_page(doc_id, name, page)
            path = hit.get("image_path", "")
            key = (doc_id, str(path))
            if not path or key in image_seen:
                continue
            image_seen.add(key)
            refs["images"].append({
                "document_id": doc_id, "document": name, "page": page,
                "image_number": hit.get("image_number"), "path": path,
                "distance": hit.get("distance_clip", hit.get("distance")),
            })

    for entry in result.get("manual_pages") or []:
        if isinstance(entry, dict):
            add_page(entry.get("document_id"), entry.get("document", "?"), entry.get("page"))
        else:
            add_page((default_document or {}).get("id"),
                     (default_document or {}).get("name", "Document actif"), entry)

    refs["images"] = refs["images"][:9]
    refs["passages"] = refs["passages"][:6]
    return refs


def show_sources(refs: dict, docs_by_id: dict, selected: dict | None) -> None:
    with st.expander("📚 Sources et outils utilisés"):
        if selected:
            st.caption("Recherche ciblée : " + selected["name"])
        else:
            st.caption("Recherche globale : tous les documents indexés")
        pages = refs.get("pages") or []
        if pages:
            st.write("**Pages retrouvées :**")
            for item in pages:
                st.write(f"- {item['document']} — page {item['page']}")
        else:
            st.write("Aucune page retrouvée")
        st.write("**Outils exécutés :** " + " → ".join(refs.get("tools") or ["aucun"]))
        st.caption("Ces références proviennent des outils et ne prouvent pas, à elles seules, chaque affirmation.")

        images = refs.get("images") or []
        if images:
            st.write("**Schémas et photographies des manuels :**")
            valid = []
            for img in images:
                path = safe_image_path(img.get("path", ""), docs_by_id.get(img.get("document_id")))
                if path:
                    valid.append((img, path))
            if valid:
                columns = st.columns(min(3, len(valid)))
                for i, (img, path) in enumerate(valid):
                    cap = f"{img['document']} — page {img['page']}, image {img['image_number']}"
                    if isinstance(img.get("distance"), (float, int)):
                        cap += f" · CLIP {img['distance']:.4f}"
                    with columns[i % len(columns)]:
                        st.image(str(path), caption=cap, use_container_width=True)
        passages = refs.get("passages") or []
        if passages:
            st.write("**Extraits consultés :**")
            for item in passages:
                title = item["document"]
                if item.get("page") is not None:
                    title += f" — page {item['page']}"
                st.caption(title)
                st.code(item["text"], language=None)


def display_message(message: dict, docs_by_id: dict, selected: dict | None) -> None:
    with st.chat_message(message.get("role", "assistant")):
        st.markdown(message.get("text", ""))
        if message.get("role") == "user" and message.get("photo_bytes"):
            st.image(message["photo_bytes"], caption="Photographie transmise", width=260)
        if message.get("role") == "assistant" and message.get("sources"):
            show_sources(message["sources"], docs_by_id, selected)


def validate_photo(uploaded) -> tuple[bytes, str]:
    raw = uploaded.getvalue()
    if not raw or len(raw) > MAX_PHOTO_BYTES:
        raise ValueError("Photo vide ou de plus de 12 Mo.")
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in PHOTO_EXTENSIONS:
        raise ValueError("Format de photo attendu : JPEG, PNG ou WebP.")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError("Photo invalide.") from exc
    return raw, suffix


def invoke_agent(selected: dict | None, question: str, photo_bytes: bytes | None,
                 suffix: str, text_k: int, image_k: int) -> dict:
    temporary = None
    try:
        if photo_bytes is not None:
            with tempfile.NamedTemporaryFile(prefix="maintidoc_photo_", suffix=suffix,
                                             delete=False) as file:
                file.write(photo_bytes)
                temporary = Path(file.name)
        photo = str(temporary) if temporary else None
        if selected is None:
            return query_all_documents(question, photo, text_k, image_k)
        return query_agent(selected, question, photo, text_k, image_k)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


st.title("🔧 MaintiDoc")
st.caption("Assistant RAG agentique multimodal — recherche ciblée ou globale")

if "active_document" not in st.session_state:
    st.session_state.active_document = "pacific"
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

available = list_ready_documents()
by_id = {doc["id"]: doc for doc in available}
if not available:
    st.error("Aucun PDF indexé : commence par exécuter uv run python main.py.")
    st.stop()
if st.session_state.active_document not in by_id:
    st.session_state.active_document = available[0]["id"]

with st.sidebar:
    st.header("Documents")
    scope = st.radio(
        "Périmètre de recherche",
        options=["all", "single"],
        format_func=lambda value: "Tous les documents" if value == "all" else "Un seul document",
        key="search_scope",
        horizontal=False,
    )
    selected = None
    if scope == "single":
        ids = list(by_id)
        choice = st.selectbox(
            "Document actif", options=ids,
            index=ids.index(st.session_state.active_document),
            format_func=lambda ident: ("PACIFIC — " if ident == "pacific" else "") + by_id[ident]["name"],
        )
        st.session_state.active_document = choice
        selected = by_id[choice]
        st.success("Recherche limitée à : " + selected["name"])
    else:
        st.success(f"Recherche dans {len(available)} document(s) indexé(s)")
        st.caption("Les passages des manuels sont comparés, puis cités par nom et par page.")

    st.divider()
    st.subheader("Ajouter un PDF technique")
    st.caption("PDF avec texte, tableau et image extractibles — maximum 25 Mo")
    upload = st.file_uploader("Importer un PDF", type=["pdf"], key="uploaded_pdf")
    if upload is not None:
        raw = upload.getvalue()
        if len(raw) > MAX_PDF_SIZE:
            st.error("Limite de taille : 25 Mo")
        else:
            try:
                import pymupdf
                with pymupdf.open(stream=raw, filetype="pdf") as pdf:
                    st.caption(f"{upload.name} — {len(pdf)} page(s)")
            except Exception:
                st.warning("Impossible de prévisualiser ce PDF.")
            if st.button("Indexer ce PDF", type="primary", use_container_width=True):
                try:
                    doc = register_upload(upload.name, raw)
                    with st.spinner("Extraction, descriptions, embeddings et vérification..."):
                        index_document(doc)
                    st.session_state.active_document = doc["id"]
                    st.success("Document indexé. Il est maintenant inclus dans la recherche globale.")
                    st.rerun()
                except Exception as exc:
                    st.error("Indexation impossible : " + str(exc))
                    st.caption("Les collections déjà indexées n'ont pas été effacées.")

    st.divider()
    st.subheader("Recherche")
    text_k = st.slider("Nombre global de passages BGE-M3", 2, 10, 5)
    image_k = st.slider("Nombre global d'images CLIP", 1, 6, 3)
    conv_key = "all" if scope == "all" else "doc:" + selected["id"]
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.conversations[conv_key] = []
        st.rerun()

if selected is not None and not is_ready(selected):
    st.error("Index du manuel sélectionné indisponible.")
    st.stop()

messages = st.session_state.conversations.setdefault(conv_key, [])
if not messages:
    st.info("Pose une question sur tous les manuels ou sur le document ciblé. Tu peux ajouter une photo.")
for message in messages:
    display_message(message, by_id, selected)

placeholder = ("Question sur tous les documents indexés..." if selected is None
               else f"Question sur {selected['name']}...")
submission = st.chat_input(placeholder, accept_file=True, file_type=["jpg", "jpeg", "png", "webp"])
if submission is not None:
    question = (submission.text or "").strip()
    files = list(submission.files or [])
    if not question and files:
        question = "Que montre cette photographie, selon les documents disponibles ?"
    if not question:
        st.warning("Écris une question ou ajoute une photographie.")
        st.stop()
    photo_bytes, suffix = None, ""
    if files:
        try:
            photo_bytes, suffix = validate_photo(files[0])
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    user_message = {"role": "user", "text": question, "photo_bytes": photo_bytes}
    messages.append(user_message)
    display_message(user_message, by_id, selected)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche et génération avec l'agent..."):
                result = invoke_agent(selected, question, photo_bytes, suffix, text_k, image_k)
            answer = str(result.get("answer") or "Aucune réponse obtenue.")
            sources = summarize_trace(result, selected)
            st.markdown(answer)
            show_sources(sources, by_id, selected)
            messages.append({"role": "assistant", "text": answer, "sources": sources})
        except Exception as exc:
            st.error("Erreur de l'agent : " + str(exc))
            st.caption("Vérifie Ollama et les modèles Qwen installés.")
    st.session_state.conversations[conv_key] = messages[-24:]
