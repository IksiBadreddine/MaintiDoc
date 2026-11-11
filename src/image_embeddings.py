"""Embeddings images CLIP, reconstruction de collection avec --rebuild."""
import argparse
import json
from collections import Counter
import chromadb
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from config import DOCUMENT_DIR, MANIFEST_PATH, CHROMA_PATH, IMAGE_COLLECTION, IMAGE_MODEL


def main(rebuild=False):
    elements = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    images = [e for e in elements if e['type'] == 'image']
    if not images:
        raise ValueError('Aucune image dans le manifeste')
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    existing = [c.name for c in client.list_collections()]
    if IMAGE_COLLECTION in existing:
        if not rebuild:
            raise RuntimeError(f'Collection {IMAGE_COLLECTION} déjà présente. Relancer avec --rebuild pour la remplacer.')
        client.delete_collection(IMAGE_COLLECTION)
    collection = client.create_collection(IMAGE_COLLECTION, embedding_function=None)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(IMAGE_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
    failures = []
    for e in images:
        path = DOCUMENT_DIR / e['content']
        try:
            with Image.open(path) as im:
                inputs = processor(images=im.convert('RGB'), return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
                if hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            md = e['metadata']
            collection.upsert(
                ids=[f'{DOCUMENT_DIR.name}_page_{md["page"]}_image_{md["image_number"]}'],
                embeddings=[features.squeeze(0).cpu().tolist()],
                metadatas=[{**md, 'type': 'image', 'image_path': str(path), 'model': IMAGE_MODEL}],
                documents=[str(path)],
            )
        except Exception as exc:
            failures.append((str(path), str(exc)))
    print(f'{collection.name}: {collection.count()} vecteurs / {len(images)} images ; erreurs : {len(failures)}')
    for path, error in failures[:10]:
        print(f'ERREUR {path}: {error}')
    if failures or collection.count() != len(images):
        raise RuntimeError('Index images incomplet : corriger les erreurs avant le RAG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    main(parser.parse_args().rebuild)
