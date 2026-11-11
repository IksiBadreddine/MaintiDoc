"""Vérifie fichiers, métadonnées, dimensions et nombre de vecteurs."""
import json
from collections import Counter
import chromadb
from config import DOCUMENT_DIR, MANIFEST_PATH, CHROMA_PATH, TEXT_COLLECTION, IMAGE_COLLECTION


def main():
    elements = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    counts = Counter(e['type'] for e in elements)
    assert set(counts).issubset({'text', 'table', 'image'})
    for e in elements:
        md = e['metadata']
        assert md['source'] and isinstance(md['page'], int) and md['page'] >= 1, e
        if e['type'] == 'image':
            assert (DOCUMENT_DIR / e['content']).is_file(), e['content']
        if e['type'] == 'table':
            assert isinstance(e.get('rows'), list), 'Tableau sans rows (réextraction nécessaire)'
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    for name, dim in ((TEXT_COLLECTION, 1024), (IMAGE_COLLECTION, 512)):
        collection = client.get_collection(name)
        data = collection.get(include=['metadatas', 'embeddings'], limit=1)
        total = collection.count()
        assert total > 0, f'{name} est vide'
        assert len(data['embeddings'][0]) == dim, f'Dimension incorrecte dans {name}'
        print(f'{name}: {total} vecteurs, dimension {dim}, exemple : {data["metadatas"][0]}')
        if name == IMAGE_COLLECTION:
            assert total == counts['image'], 'Nombre de vecteurs images différent du manifeste'
    print(f'Manifeste: {len(elements)} éléments, types: {dict(counts)}')
    print('Vérifications structurelles OK. La qualité visuelle et sémantique reste à contrôler.')


if __name__ == '__main__':
    main()
