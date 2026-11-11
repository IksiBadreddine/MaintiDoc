import argparse
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import CHROMA_PATH, TEXT_COLLECTION, TEXT_MODEL


def main(question):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=TEXT_COLLECTION)
    index = VectorStoreIndex.from_vector_store(
        vector_store=ChromaVectorStore(chroma_collection=collection),
        embed_model=HuggingFaceEmbedding(model_name=TEXT_MODEL),
    )
    for i, result in enumerate(index.as_retriever(similarity_top_k=3).retrieve(question), 1):
        print(f'\nRésultat {i} — score={result.score:.4f} page={result.metadata.get("page")} type={result.metadata.get("type")}')
        print(result.text[:1000])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('question', nargs='?', default='Quelles sont les causes possibles si le compresseur ne démarre pas ?')
    main(parser.parse_args().question)
