
import chromadb
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_PATH))

for name in ["maintenance_pacific", "maintenance_images"]:
    collection = client.get_collection(name=name)
    data = collection.get(include=["metadatas"])

    counts = Counter(
        metadata.get("type", "unknown")
        for metadata in data["metadatas"]
    )

    print(f"\nCollection : {name}")
    print(f"Vecteurs : {collection.count()}")
    print(f"Types : {dict(counts)}")
    print("Exemple de métadonnées :", data["metadatas"][0])
