
"""Recherche d'images du manuel à partir d'une photo utilisateur."""

import argparse
from pathlib import Path

import chromadb
import torch
from PIL import Image

from config import CHROMA_PATH, IMAGE_COLLECTION
from visual_search import load_clip


def search_by_photo(photo_path: str, top_k: int = 5) -> list[dict]:
    """Recherche les images proches d'une photo dans ChromaDB."""

    path = Path(photo_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Photo introuvable : {path}")

    if top_k < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    # 1. Charger le modèle CLIP déjà utilisé
    model, processor, device = load_clip()

    # 2. Préparer la photo utilisateur
    with Image.open(path) as image:
        inputs = processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 3. Générer un embedding visuel de 512 dimensions
    with torch.inference_mode():
        features = model.get_image_features(**inputs)

        if hasattr(features, "pooler_output"):
            features = features.pooler_output

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=-1,
        )

    # 4. Interroger les images déjà indexées
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=IMAGE_COLLECTION
    )

    if collection.count() == 0:
        raise RuntimeError("Collection d'images vide.")

    results = collection.query(
        query_embeddings=[features[0].cpu().tolist()],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    # 5. Organiser les résultats
    hits = []

    for rank, (metadata, distance) in enumerate(
        zip(
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        metadata = metadata or {}

        hit = {
            "rank": rank,
            "page": metadata.get("page"),
            "image_number": metadata.get("image_number"),
            "image_path": metadata.get("image_path"),
            "distance": float(distance),
        }

        hits.append(hit)

        print(f"\nImage {rank}")
        print(f"Page : {hit['page']}")
        print(f"Numéro : {hit['image_number']}")
        print(f"Fichier : {hit['image_path']}")
        print(f"Distance : {hit['distance']:.4f}")

    return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recherche visuelle par photographie"
    )

    parser.add_argument("photo", type=str)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    search_by_photo(
        args.photo,
        top_k=args.top_k,
    )
