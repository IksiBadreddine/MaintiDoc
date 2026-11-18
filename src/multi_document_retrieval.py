"""Recherche multi-PDF dans des collections ChromaDB déjà indexées.

IMPORTANT : les scores BGE-M3 ne sont comparés qu'entre collections de
même modèle et même métrique ChromaDB ; jamais aux distances CLIP.
Aucun document n'est réindexé lors de la recherche.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _text_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from config import TEXT_MODEL
    return HuggingFaceEmbedding(model_name=TEXT_MODEL)


@lru_cache(maxsize=1)
def _db():
    import chromadb
    from config import CHROMA_PATH
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def _rank_results(results: list[dict], top_k: int) -> list[dict]:
    """Classe les candidats globalement par distance, sans fusion BGE/CLIP."""
    if top_k < 1:
        raise ValueError("top_k doit être >= 1")
    unique = {}
    for result in results:
        key = (result["document_id"], result["vector_id"])
        if key not in unique or result["distance"] < unique[key]["distance"]:
            unique[key] = result
    return sorted(unique.values(), key=lambda item: item["distance"])[:top_k]


def _query_collections(docs: list[dict], kind: str, vector: list[float],
                       top_k: int, *, candidate_factor: int = 3) -> list[dict]:
    if not docs:
        return []
    if kind not in {"text", "image"}:
        raise ValueError("Type de recherche invalide")
    collection_key = "text_collection" if kind == "text" else "image_collection"
    results = []
    client = _db()
    for doc in docs:
        col = client.get_collection(name=doc[collection_key])
        count = col.count()
        if not count:
            continue
        data = col.query(
            query_embeddings=[vector],
            n_results=min(max(top_k * candidate_factor, 5), count),
            include=["documents", "metadatas", "distances"],
        )
        ids = (data.get("ids") or [[]])[0]
        metas = (data.get("metadatas") or [[]])[0]
        contents = (data.get("documents") or [[]])[0]
        distances = (data.get("distances") or [[]])[0]
        for uid, meta, content, distance in zip(ids, metas, contents, distances):
            meta = meta or {}
            hit = {
                "document_id": doc["id"],
                "document": doc["name"],
                "vector_id": str(uid),
                "page": meta.get("page", "?"),
                "type": meta.get("type", kind),
                "distance": float(distance),
            }
            if kind == "image":
                hit["image_number"] = meta.get("image_number")
                hit["image_path"] = meta.get("image_path", "")
            else:
                hit["content"] = str(content or "")
                hit["image_number"] = meta.get("image_number")
            results.append(hit)
    return _rank_results(results, top_k)


def search_text(docs: list[dict], query: str, top_k: int) -> list[dict]:
    """Une seule vectorisation BGE-M3, puis recherche dans toutes les collections."""
    if not query.strip():
        return []
    embedding = _text_model().get_query_embedding(query)
    # Interroger un ensemble de candidats plus large permet d'éviter qu'une
    # description d'image erronée prenne la place du texte technique original.
    candidates = _query_collections(docs, "text", list(embedding), max(3 * top_k, top_k))
    originals = [h for h in candidates if h["type"] in {"text", "table"} and h["content"].strip()]
    captions = [h for h in candidates if h["type"] == "image_description" and h["content"].strip()]
    return (originals + captions)[:top_k]


def _clip_vector(*, query: str | None = None, photo: str | None = None) -> list[float]:
    import torch
    from PIL import Image
    from visual_search import load_clip
    if (query is None) == (photo is None):
        raise ValueError("Fournir une question OU une photographie")
    model, processor, device = load_clip()
    if photo is not None:
        path = Path(photo).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Photographie introuvable : {path}")
        with Image.open(path) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        feature_method = model.get_image_features
    else:
        inputs = processor(text=[query], padding=True, truncation=True,
                           max_length=77, return_tensors="pt")
        feature_method = model.get_text_features
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = feature_method(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        if not isinstance(features, torch.Tensor):
            raise TypeError("Embedding CLIP invalide")
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features[0].cpu().tolist()


def search_images(docs: list[dict], *, top_k: int = 3,
                  query: str | None = None, photo: str | None = None) -> list[dict]:
    if query is not None and not query.strip():
        return []
    embedding = _clip_vector(query=query, photo=photo)
    return _query_collections(docs, "image", embedding, top_k)


@lru_cache(maxsize=32)
def _manifest(path_str: str) -> list[dict[str, Any]]:
    with open(path_str, "r", encoding="utf-8") as src:
        elements = json.load(src)
    if not isinstance(elements, list):
        raise ValueError("Manifeste invalide")
    return elements


def original_page(doc: dict, page: int, max_chars: int = 5000) -> dict:
    """Texte et tableaux du PDF désigné, sans mélange inter-document."""
    if not isinstance(page, int) or not (1 <= page <= 1000):
        return {"document_id": doc["id"], "document": doc["name"], "page": page,
                "context": "", "found": False}
    manifest = doc["document_dir"] / "processed" / "extracted_data.json"
    elements = _manifest(str(manifest))
    out = []
    for item in elements:
        if item.get("type") not in {"text", "table"}:
            continue
        if str((item.get("metadata") or {}).get("page")) != str(page):
            continue
        value = item.get("content")
        if not isinstance(value, str) or not value.strip():
            continue
        out.append(f"[{item['type']} original] {value.strip()}")
    return {"document_id": doc["id"], "document": doc["name"], "page": page,
            "context": "\n\n".join(out)[:max_chars], "found": bool(out)}


def image_caption(doc: dict, page: Any, number: Any, max_chars: int = 1000) -> str:
    manifest = doc["document_dir"] / "processed" / "extracted_data.json"
    for item in _manifest(str(manifest)):
        if item.get("type") != "image":
            continue
        meta = item.get("metadata") or {}
        if str(meta.get("page")) == str(page) and str(meta.get("image_number")) == str(number):
            return str(item.get("description", ""))[:max_chars]
    return ""


def format_sources(hits: list[dict], max_chars: int = 10000) -> str:
    """Le nom du PDF figure AVANT chaque citation, même si la page est identique."""
    sections = []
    for position, hit in enumerate(hits, 1):
        sections.append(
            f"[SOURCE {position}] DOCUMENT : {hit['document']} "
            f"(id: {hit['document_id']}); PAGE : {hit['page']}; "
            f"TYPE : {hit['type']}\n{hit.get('content', '')[:1800]}"
        )
    return "\n\n---\n\n".join(sections)[:max_chars]
