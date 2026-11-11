"""Index textuel : reconstruction explicite avec --rebuild."""
import argparse
import json
import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import MANIFEST_PATH, CHROMA_PATH, TEXT_COLLECTION, TEXT_MODEL


def main(rebuild=False):
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f'Manifeste absent : {MANIFEST_PATH}')
    elements = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    documents = [Document(text=e['content'], metadata={**e['metadata'], 'type': e['type']})
                 for e in elements if e['type'] in ('text', 'table') and e['content'].strip()]
    if not documents:
        raise ValueError('Aucun texte/tableau à indexer')
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    existing = [c.name for c in client.list_collections()]
    if TEXT_COLLECTION in existing:
        if not rebuild:
            raise RuntimeError(f'Collection {TEXT_COLLECTION} déjà présente. Relancer avec --rebuild pour la remplacer.')
        client.delete_collection(TEXT_COLLECTION)
    collection = client.create_collection(TEXT_COLLECTION)
    store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=HuggingFaceEmbedding(model_name=TEXT_MODEL),
        transformations=[SentenceSplitter(chunk_size=500, chunk_overlap=80)],
        show_progress=True,
    )
    print(f'{collection.name}: {collection.count()} vecteurs (dimension attendue 1024)')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    main(parser.parse_args().rebuild)
