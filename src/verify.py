"""Vérification globale du pipeline MaintiDoc."""

import json
from collections import Counter

import chromadb

from config import (
    DOCUMENT_DIR,
    MANIFEST_PATH,
    CHROMA_PATH,
    TEXT_COLLECTION,
    IMAGE_COLLECTION,
)


def main():

    # -----------------------------------
    # 1. VÉRIFICATION DU MANIFESTE
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

    assert isinstance(elements, list)

    counts = Counter(
        element["type"]
        for element in elements
    )

    assert set(counts).issubset({
        "text",
        "table",
        "image",
    })

    image_paths = set()

    for element in elements:

        metadata = element["metadata"]

        assert metadata["source"]

        assert (
            isinstance(metadata["page"], int)
            and metadata["page"] >= 1
        )

        if element["type"] == "table":

            assert isinstance(
                element.get("rows"),
                list,
            ), "Tableau sans rows."

        elif element["type"] == "image":

            image_path = (
                DOCUMENT_DIR / element["content"]
            )

            assert image_path.is_file(), (
                f"Image absente : {image_path}"
            )

            assert isinstance(
                element.get("context"),
                str,
            ), "Contexte d'image absent."

            description = element.get(
                "description", ""
            )

            assert (
                isinstance(description, str)
                and description.strip()
            ), (
                f"Description absente : {image_path}"
            )

            assert (
                element.get("caption_status")
                == "completed"
            ), (
                f"Captioning incomplet : {image_path}"
            )

            assert element.get(
                "caption_model"
            ), (
                f"Modèle de captioning absent : {image_path}"
            )

            image_paths.add(
                element["content"]
            )

    assert len(image_paths) == counts["image"], (
        "Chemins d'images dupliqués."
    )

    print("\nMANIFESTE VÉRIFIÉ")
    print(f"Textes : {counts['text']}")
    print(f"Tableaux : {counts['table']}")
    print(f"Images : {counts['image']}")

    print(
        "Descriptions valides : "
        f"{len(image_paths)}"
    )

    # -----------------------------------
    # 2. CHROMADB
    # -----------------------------------

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    # -----------------------------------
    # 3. DIMENSIONS DES COLLECTIONS
    # -----------------------------------

    for name, expected_dimension in (
        (TEXT_COLLECTION, 1024),
        (IMAGE_COLLECTION, 512),
    ):

        collection = client.get_collection(
            name
        )

        total = collection.count()

        assert total > 0, (
            f"Collection vide : {name}"
        )

        sample = collection.get(
            include=[
                "metadatas",
                "embeddings",
            ],
            limit=1,
        )

        embeddings = sample["embeddings"]

        assert embeddings is not None
        assert len(embeddings) > 0

        dimension = len(embeddings[0])

        assert dimension == expected_dimension, (
            f"Dimension incorrecte dans {name} : "
            f"{dimension} au lieu de "
            f"{expected_dimension}"
        )

        print(f"\nCollection : {name}")
        print(f"Vecteurs : {total}")
        print(f"Dimension : {dimension}")

        if name == IMAGE_COLLECTION:

            assert total == counts["image"], (
                "Nombre de vecteurs CLIP différent "
                "du nombre d'images."
            )

    # -----------------------------------
    # 4. DESCRIPTIONS DANS CHROMADB
    # -----------------------------------

    text_collection = client.get_collection(
        TEXT_COLLECTION
    )

    descriptions = text_collection.get(
        where={
            "type": "image_description"
        },
        include=[
            "metadatas"
        ],
    )

    description_metadatas = (
        descriptions["metadatas"] or []
    )

    indexed_paths = {
        metadata["image_path"]
        for metadata in description_metadatas
        if (
            metadata is not None
            and "image_path" in metadata
        )
    }

    missing_paths = (
        image_paths - indexed_paths
    )

    unexpected_paths = (
        indexed_paths - image_paths
    )

    assert not missing_paths, (
        "Descriptions non indexées : "
        f"{sorted(missing_paths)}"
    )

    assert not unexpected_paths, (
        "Descriptions d'images inconnues : "
        f"{sorted(unexpected_paths)}"
    )

    print(
        "\nINDEXATION DES DESCRIPTIONS"
    )

    print(
        "Images avec description indexée : "
        f"{len(indexed_paths)}"
    )

    print(
        "Vecteurs de descriptions : "
        f"{len(description_metadatas)}"
    )

    # -----------------------------------
    # 5. BILAN FINAL
    # -----------------------------------

    print("\nVÉRIFICATIONS TERMINÉES")

    print(
        "Extraction : OK\n"
        "Descriptions : OK\n"
        "Embeddings BGE-M3 : OK\n"
        "Embeddings CLIP : OK\n"
        "Indexation des descriptions : OK"
    )

    print(
        "\nNote : ces contrôles valident la "
        "structure et l'indexation, pas "
        "l'exactitude sémantique des descriptions."
    )


if __name__ == "__main__":
    main()