
"""RAG multimodal MaintiDoc : BGE-M3 (textes) + CLIP (images) + Qwen3.

Les images retrouvées par CLIP sont représentées dans le contexte du LLM
par les descriptions déjà présentes dans le manifeste d'extraction.
"""

import argparse
import json
from pathlib import Path

from config import MANIFEST_PATH
from rag import build_context, generate_answer, load_retriever
from visual_search import search_images


def image_key(page, image_number) -> tuple[str, str]:
    """Identifie une image par sa page et son numéro dans le manuel."""
    return str(page), str(image_number)


def load_image_descriptions() -> dict[tuple[str, str], str]:
    """Charge les descriptions Qwen-VL déjà produites (sans relancer le modèle)."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifeste absent : {MANIFEST_PATH}")

    elements = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(elements, list):
        raise ValueError("Le manifeste doit contenir une liste d'éléments.")

    descriptions = {}
    for element in elements:
        if element.get("type") != "image":
            continue
        if element.get("caption_status") != "completed":
            continue
        metadata = element.get("metadata") or {}
        description = element.get("description", "")
        if not isinstance(description, str) or not description.strip():
            continue
        key = image_key(metadata.get("page"), metadata.get("image_number"))
        descriptions[key] = description.strip()
    return descriptions


def run(question: str, text_k: int = 5, image_k: int = 3) -> dict:
    """Recherche dans les deux collections et génère une réponse avec Qwen3."""
    if not question.strip():
        raise ValueError("La question ne peut pas être vide.")
    if text_k < 1 or image_k < 1:
        raise ValueError("text_k et image_k doivent être supérieurs ou égaux à 1.")

    # 1. Recherche BGE-M3 : textes, tableaux et descriptions d'images.
    print("Chargement du retriever BGE-M3...")
    retriever = load_retriever(top_k=text_k)
    print("Recherche des passages textuels...")
    text_results = retriever.retrieve(question)
    text_context, text_sources = build_context(text_results)
    print(f"{len(text_sources)} passages textuels récupérés.")

    # 2. Recherche CLIP : représentation visuelle des images du manuel.
    print("Recherche des images avec CLIP...")
    image_hits = search_images(question, top_k=image_k, verbose=False)
    print(f"{len(image_hits)} images retrouvées.")

    # 3. Récupération des captions existantes et déduplication.
    descriptions = load_image_descriptions()
    already_in_text = set()
    for result in text_results:
        metadata = result.node.metadata or {}
        if metadata.get("type") == "image_description":
            already_in_text.add(
                image_key(metadata.get("page"), metadata.get("image_number"))
            )

    visual_passages = []
    for hit in image_hits:
        key = image_key(hit["page"], hit["image_number"])
        description = descriptions.get(key)
        if not description or key in already_in_text:
            continue
        visual_passages.append(
            f"[Image CLIP {hit['rank']}]\n"
            f"Page : {hit['page']}\n"
            f"Numéro de l'image : {hit['image_number']}\n"
            f"Description extraite de l'image :\n{description}"
        )
        already_in_text.add(key)

    # 4. Fusion des CONTEXTES, et non des scores vectoriels.
    context_parts = []
    if text_context:
        context_parts.append(text_context)
    if visual_passages:
        context_parts.append("\n\n---\n\n".join(visual_passages))
    context = "\n\n---\n\n".join(context_parts)

    if not context:
        print("Aucun contexte exploitable trouvé.")
        return {"answer": "", "text_sources": text_sources, "images": image_hits}

    # 5. Génération avec le Qwen3 déjà branché dans rag.py.
    print("Génération de la réponse avec Qwen3...\n")
    answer = generate_answer(question, context)

    print("=" * 65)
    print("RÉPONSE MULTIMODALE MAINTIDOC")
    print("=" * 65)
    print(answer)

    print("\n" + "=" * 65)
    print("PASSAGES BGE-M3")
    print("=" * 65)
    for position, source in enumerate(text_sources, start=1):
        score = source.get("score")
        score_text = f"{score:.4f}" if score is not None else "N/A"
        print(
            f"{position}. Page {source['page']} | "
            f"Type : {source['type']} | Score : {score_text}"
        )

    print("\n" + "=" * 65)
    print("IMAGES RETROUVÉES PAR CLIP")
    print("=" * 65)
    for hit in image_hits:
        path = Path(hit["image_path"]) if hit["image_path"] else None
        exists = path.is_file() if path else False
        print(
            f"{hit['rank']}. Page {hit['page']} | "
            f"Image {hit['image_number']} | "
            f"Distance {hit['distance']:.4f}"
        )
        print(f"   Fichier : {path}")
        print(f"   Existe : {'oui' if exists else 'non'}")

    return {"answer": answer, "text_sources": text_sources, "images": image_hits}


def main():
    parser = argparse.ArgumentParser(description="RAG multimodal MaintiDoc")
    parser.add_argument("question", nargs="+", help="Question sur le manuel")
    parser.add_argument("--text-k", type=int, default=5)
    parser.add_argument("--image-k", type=int, default=3)
    args = parser.parse_args()
    if args.text_k < 1 or args.image_k < 1:
        parser.error("--text-k et --image-k doivent être supérieurs ou égaux à 1.")
    run(" ".join(args.question), text_k=args.text_k, image_k=args.image_k)


if __name__ == "__main__":
    main()
