"""Extraction des textes, tableaux et images du PDF MaintiDoc."""

import argparse
import json
import shutil
from collections import Counter

import pymupdf
from llama_index.core import SimpleDirectoryReader

from config import PDF_PATH, DOCUMENT_DIR, MANIFEST_PATH


def write_json(path, data):
    """Enregistre un objet au format JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rectangle_distance(rect_a, rect_b):
    """Distance approximative entre deux rectangles PDF."""
    dx = max(
        rect_a.x0 - rect_b.x1,
        rect_b.x0 - rect_a.x1,
        0,
    )

    dy = max(
        rect_a.y0 - rect_b.y1,
        rect_b.y0 - rect_a.y1,
        0,
    )

    return dx + dy


def get_image_context(page, xref, max_chars=1500):
    """
    Récupère le texte proche d'une image.

    Priorité :
    1. Blocs textuels proches de l'image.
    2. Texte général de la page si aucune position
       exploitable n'est disponible.

    La proximité géométrique est une heuristique :
    elle ne garantit pas que le texte décrit l'image.
    """

    image_rects = page.get_image_rects(xref)

    text_blocks = []

    for block in page.get_text("blocks"):
        if len(block) < 7:
            continue

        # 0 = bloc de texte ; 1 = bloc d'image
        if block[6] != 0:
            continue

        text = block[4].strip()

        if not text:
            continue

        text_blocks.append({
            "rect": pymupdf.Rect(block[:4]),
            "text": text,
        })

    if not text_blocks:
        return "", "none"

    # Si la position de l'image n'est pas disponible,
    # utiliser le texte général de la page.
    if not image_rects:
        context = "\n".join(
            block["text"] for block in text_blocks
        )
        return context[:max_chars], "page_fallback"

    candidates = []

    for block in text_blocks:
        distance = min(
            rectangle_distance(
                block["rect"],
                image_rect,
            )
            for image_rect in image_rects
        )

        candidates.append(
            (distance, block["text"])
        )

    candidates.sort(key=lambda item: item[0])

    # Sélection des blocs proches de l'image.
    nearby = [
        text
        for distance, text in candidates
        if distance <= 200
    ][:4]

    if nearby:
        return "\n".join(nearby)[:max_chars], "nearby_blocks"

    # Repli : les trois blocs les plus proches.
    closest = [
        text
        for _, text in candidates[:3]
    ]

    return "\n".join(closest)[:max_chars], "closest_blocks"


def main(rebuild=False):

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF introuvable : {PDF_PATH}"
        )

    # Nettoyage uniquement des artefacts d'extraction.
    # Le cache des descriptions et ChromaDB sont préservés.
    if rebuild:
        for name in (
            "text",
            "tables",
            "images",
            "processed",
        ):
            shutil.rmtree(
                DOCUMENT_DIR / name,
                ignore_errors=True,
            )

    # Copie du PDF original.
    original = (
        DOCUMENT_DIR
        / "original"
        / PDF_PATH.name
    )

    original.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(PDF_PATH, original)

    elements = []

    text_dir = DOCUMENT_DIR / "text"
    table_dir = DOCUMENT_DIR / "tables"
    image_dir = DOCUMENT_DIR / "images"

    # -----------------------------------
    # 1. EXTRACTION DES TEXTES
    # -----------------------------------

    documents = SimpleDirectoryReader(
        input_files=[str(PDF_PATH)]
    ).load_data()

    for doc in documents:

        page_number = int(
            doc.metadata["page_label"]
        )

        content = doc.text.strip()

        if not content:
            continue

        item = {
            "type": "text",
            "content": content,
            "metadata": {
                "source": PDF_PATH.name,
                "page": page_number,
            },
        }

        elements.append(item)

        write_json(
            text_dir / f"page_{page_number:03d}.json",
            item,
        )

    # -----------------------------------
    # 2. EXTRACTION TABLEAUX ET IMAGES
    # -----------------------------------

    with pymupdf.open(PDF_PATH) as pdf:

        for page_number, page in enumerate(
            pdf,
            start=1,
        ):

            # TABLEAUX
            tables = page.find_tables().tables

            for table_number, table in enumerate(
                tables,
                start=1,
            ):

                rows = [
                    [
                        str(cell).strip()
                        if cell is not None
                        else ""
                        for cell in row
                    ]
                    for row in table.extract()
                ]

                content = "\n".join(
                    " | ".join(row)
                    for row in rows
                )

                if not content.strip():
                    continue

                item = {
                    "type": "table",
                    "content": content,
                    "rows": rows,
                    "metadata": {
                        "source": PDF_PATH.name,
                        "page": page_number,
                        "table_number": table_number,
                    },
                }

                elements.append(item)

                write_json(
                    table_dir
                    / (
                        f"page_{page_number:03d}"
                        f"_table_{table_number:02d}.json"
                    ),
                    item,
                )

            # IMAGES
            images = page.get_images(full=True)

            for image_number, info in enumerate(
                images,
                start=1,
            ):

                xref = info[0]

                extracted = pdf.extract_image(xref)

                if (
                    not extracted
                    or not extracted.get("image")
                ):
                    continue

                extension = extracted["ext"]

                filename = (
                    f"page_{page_number:03d}"
                    f"_image_{image_number:02d}"
                    f".{extension}"
                )

                image_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                image_path = image_dir / filename

                image_path.write_bytes(
                    extracted["image"]
                )

                # NOUVEAU : contexte propre à l'image.
                context, context_source = (
                    get_image_context(page, xref)
                )

                item = {
                    "type": "image",
                    "content": f"images/{filename}",
                    "context": context,
                    "context_source": context_source,
                    "metadata": {
                        "source": PDF_PATH.name,
                        "page": page_number,
                        "image_number": image_number,
                    },
                }

                elements.append(item)

    # -----------------------------------
    # 3. MANIFESTE GLOBAL
    # -----------------------------------

    write_json(MANIFEST_PATH, elements)

    counts = Counter(
        element["type"]
        for element in elements
    )

    context_counts = Counter(
        element.get("context_source", "none")
        for element in elements
        if element["type"] == "image"
    )

    print("\nEXTRACTION TERMINÉE")
    print(f"Manifeste : {MANIFEST_PATH}")
    print(f"Textes : {counts['text']}")
    print(f"Tableaux : {counts['table']}")
    print(f"Images : {counts['image']}")
    print(f"Total : {len(elements)}")
    print(f"Sources des contextes : {dict(context_counts)}")

    print(
        "\nAttention : certains tableaux peuvent "
        "également apparaître dans le texte des pages."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruire les artefacts d'extraction.",
    )

    args = parser.parse_args()

    main(rebuild=args.rebuild)