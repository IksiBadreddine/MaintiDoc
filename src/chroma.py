"""Indexation textuelle MaintiDoc avec BGE-M3 et ChromaDB."""

import argparse
import json
from collections import Counter

import chromadb

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)

from llama_index.core.node_parser import (
    SentenceSplitter,
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)

from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)

from config import (
    MANIFEST_PATH,
    CHROMA_PATH,
    TEXT_COLLECTION,
    TEXT_MODEL,
)


EXCLUDED_EMBED_METADATA = [
    "source",
    "page",
    "type",
    "table_number",
    "image_number",
    "image_path",
    "caption_model",
]


def prepare_documents(elements):
    """Prépare les documents textuels pour LlamaIndex."""

    documents = []
    counts = Counter()

    for element in elements:

        element_type = element.get("type")

        content = element.get("content", "")

        metadata = element.get("metadata", {})

        if not isinstance(metadata, dict):
            continue

        # -----------------------------------
        # TEXTES ET TABLEAUX
        # -----------------------------------

        if element_type in ("text", "table"):

            if (
                not isinstance(content, str)
                or not content.strip()
            ):
                continue

            document = Document(
                text=content.strip(),
                metadata={
                    **metadata,
                    "type": element_type,
                },
                excluded_embed_metadata_keys=(
                    EXCLUDED_EMBED_METADATA
                ),
            )

            documents.append(document)

            counts[element_type] += 1

        # -----------------------------------
        # DESCRIPTIONS DES IMAGES
        # -----------------------------------

        elif element_type == "image":

            description = element.get(
                "description", ""
            )

            status = element.get(
                "caption_status"
            )

            if (
                not isinstance(description, str)
                or not description.strip()
                or status != "completed"
            ):
                continue

            image_metadata = {
                **metadata,
                "type": "image_description",
                "image_path": content,
            }

            if element.get("caption_model"):

                image_metadata["caption_model"] = (
                    element["caption_model"]
                )

            document = Document(
                text=description.strip(),
                metadata=image_metadata,
                excluded_embed_metadata_keys=(
                    EXCLUDED_EMBED_METADATA
                ),
            )

            documents.append(document)

            counts["image_description"] += 1

    return documents, counts


def main(rebuild=False):

    # -----------------------------------
    # 1. CHARGEMENT DU MANIFESTE
    # -----------------------------------

    if not MANIFEST_PATH.is_file():

        raise FileNotFoundError(
            f"Manifeste absent : {MANIFEST_PATH}"
        )

    elements = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(elements, list):

        raise ValueError(
            "Le manifeste doit contenir une liste."
        )

    # -----------------------------------
    # 2. PRÉPARATION
    # -----------------------------------

    documents, counts = prepare_documents(
        elements
    )

    if not documents:

        raise ValueError(
            "Aucun document à indexer."
        )

    print("\nDOCUMENTS PRÉPARÉS")

    print(f"Textes : {counts['text']}")
    print(f"Tableaux : {counts['table']}")

    print(
        "Descriptions d'images : "
        f"{counts['image_description']}"
    )

    print(f"Total : {len(documents)}")

    # -----------------------------------
    # 3. CHROMADB
    # -----------------------------------

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    existing = [
        collection.name
        for collection in client.list_collections()
    ]

    if TEXT_COLLECTION in existing:

        if not rebuild:

            raise RuntimeError(
                f"Collection {TEXT_COLLECTION} "
                "déjà présente. Utiliser --rebuild."
            )

        client.delete_collection(
            TEXT_COLLECTION
        )

    collection = client.create_collection(
        name=TEXT_COLLECTION
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # -----------------------------------
    # 4. EMBEDDINGS BGE-M3
    # -----------------------------------

    embed_model = HuggingFaceEmbedding(
        model_name=TEXT_MODEL
    )

    # -----------------------------------
    # 5. CHUNKING + INDEXATION
    # -----------------------------------

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[
            SentenceSplitter(
                chunk_size=500,
                chunk_overlap=80,
            )
        ],
        show_progress=True,
    )

    # -----------------------------------
    # 6. RÉSULTATS
    # -----------------------------------

    total_vectors = collection.count()

    print("\nINDEXATION TEXTUELLE TERMINÉE")

    print(
        f"Collection : {collection.name}"
    )

    print(
        f"Vecteurs stockés : {total_vectors}"
    )

    print(
        "Dimension attendue : 1024 (BGE-M3)"
    )

    return index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruire la collection textuelle.",
    )

    args = parser.parse_args()

    main(rebuild=args.rebuild)