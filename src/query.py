
import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

# Chargement du modèle d'embeddings
print("1. Chargement de BGE-M3...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# Connexion à la base ChromaDB existante
print("2. Connexion à ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name="maintenance_pacific")

vector_store = ChromaVectorStore(chroma_collection=collection)

# Rechargement de l'index existant
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

# Recherche sémantique sans LLM
retriever = index.as_retriever(similarity_top_k=3)

question = "Quelles sont les causes possibles si le compresseur ne démarre pas ?"

print(f"\nQuestion : {question}")
print("\n3. Résultats de la recherche :")

results = retriever.retrieve(question)

for i, result in enumerate(results, start=1):
    print(f"\n--- Résultat {i} ---")
    print(f"Score : {result.score:.4f}")
    print(f"Page : {result.metadata.get('page', 'Inconnue')}")
    print(f"Type : {result.metadata.get('type', 'Inconnu')}")
    print(f"Contenu :\n{result.text[:1000]}")
