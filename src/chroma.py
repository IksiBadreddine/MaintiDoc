
import json
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = (
    BASE_DIR
    / "data"
    / "documents"
    / "maintenance_machine"
    / "processed"
    / "extracted_data.json"
)
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

# Modèle d'embeddings
MODEL_NAME = "BAAI/bge-m3"

print("1. Chargement des données extraites...")

with JSON_PATH.open("r", encoding="utf-8") as file:
    elements = json.load(file)

# On vectorise le texte et les tableaux.
# Les images seront traitées séparément avec OCR ou vision.
documents = []

for element in elements:
    element_type = element["type"]

    if element_type not in {"text", "table"}:
        continue

    content = element["content"].strip()

    if not content:
        continue

    metadata = {
        **element["metadata"],
        "type": element_type,
    }

    documents.append(
        Document(
            text=content,
            metadata=metadata,
        )
    )

print(f"Documents à indexer : {len(documents)}")

print("\n2. Chargement de BGE-M3...")

embed_model = HuggingFaceEmbedding(
    model_name=MODEL_NAME
)

print("\n3. Initialisation de ChromaDB...")

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection_name = "maintenance_pacific"

# Supprimer l'ancien index de cette collection
try:
    client.delete_collection(name=collection_name)
    print("Ancienne collection supprimée.")
except ValueError:
    print("Aucune ancienne collection à supprimer.")

# Créer une nouvelle collection vide
collection = client.create_collection(
    name=collection_name
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# Découpage des textes longs
splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=80,
)

print("\n4. Création de l'index vectoriel...")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    transformations=[splitter],
    show_progress=True,
)

print("\n--- Indexation terminée ---")
print(f"Collection : {collection.name}")
print(f"Vecteurs enregistrés : {collection.count()}")
print(f"Dimension attendue : 1024")
print(f"Base persistante : {CHROMA_PATH}")
