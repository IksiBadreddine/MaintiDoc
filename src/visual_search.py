
"""Recherche d'images du manuel à partir d'une question textuelle avec CLIP.

Utilise la collection ChromaDB 'maintenance_images' déjà construite.
Ne recalcule pas les embeddings des images.
"""

import argparse
from functools import lru_cache

import chromadb
import torch
from transformers import CLIPModel, CLIPProcessor

from config import CHROMA_PATH, IMAGE_COLLECTION, IMAGE_MODEL


@lru_cache(maxsize=1)
def load_clip():
    """Charge une seule fois l'encodeur CLIP et son processeur."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(IMAGE_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
    return model, processor, device


def search_images(question: str, top_k: int = 5, verbose: bool = True) -> list[dict]:
    """Renvoie les images les plus proches de la question, avec leurs métadonnées."""
    if not question.strip():
        raise ValueError("La question ne peut pas être vide.")
    if top_k < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=IMAGE_COLLECTION)
    total = collection.count()
    if total == 0:
        raise RuntimeError(f"La collection {IMAGE_COLLECTION} est vide.")

    model, processor, device = load_clip()
    inputs = processor(
        text=[question],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        if not isinstance(features, torch.Tensor):
            raise TypeError("CLIP n'a pas retourné d'embedding textuel valide.")
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

    results = collection.query(
        query_embeddings=[features[0].cpu().tolist()],
        n_results=min(top_k, total),
        include=["metadatas", "distances"],
    )

    ids = (results.get("ids") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    hits = []
    for position, (vector_id, metadata, distance) in enumerate(
        zip(ids, metadatas, distances), start=1
    ):
        metadata = metadata or {}
        hits.append(
            {
                "rank": position,
                "id": vector_id,
                "page": metadata.get("page"),
                "image_number": metadata.get("image_number"),
                "image_path": metadata.get("image_path", ""),
                "distance": float(distance),
            }
        )

    if verbose:
        print("\n========== RÉSULTATS CLIP ==========")
        for hit in hits:
            print(f"\nImage {hit['rank']}")
            print(f"Page : {hit['page']}")
            print(f"Numéro : {hit['image_number']}")
            print(f"Fichier : {hit['image_path']}")
            print(f"Distance : {hit['distance']:.4f}")

    return hits


def main():
    parser = argparse.ArgumentParser(description="Recherche d'images CLIP dans MaintiDoc")
    parser.add_argument("question", nargs="+", help="Description de l'image recherchée")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k doit être supérieur ou égal à 1.")
    search_images(" ".join(args.question), top_k=args.top_k)


if __name__ == "__main__":
    main()
