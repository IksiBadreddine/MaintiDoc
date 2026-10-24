
import json
from pathlib import Path

import chromadb
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_DIR = (
    BASE_DIR / "data" / "documents" / "maintenance_machine"
)

JSON_PATH = DOCUMENT_DIR / "processed" / "extracted_data.json"
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

MODEL_NAME = "openai/clip-vit-base-patch32"
COLLECTION_NAME = "maintenance_images"

# 1. Charger les métadonnées des images
print("1. Chargement des images...")

with JSON_PATH.open("r", encoding="utf-8") as file:
    elements = json.load(file)

images = [
    element for element in elements
    if element["type"] == "image"
]

print(f"Images trouvées : {len(images)}")

# 2. Charger CLIP
print("\n2. Chargement du modèle CLIP...")

device = (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# 3. Initialiser ChromaDB
print("\n3. Initialisation de ChromaDB...")

client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,
)

# 4. Générer et stocker les embeddings
print("\n4. Indexation des images...")

indexed = 0
errors = 0

for element in images:
    metadata = element["metadata"]
    image_path = DOCUMENT_DIR / element["content"]

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            features = model.get_image_features(**inputs)

            # Compatibilité avec les versions récentes de transformers
            if hasattr(features, "pooler_output"):
                features = features.pooler_output

            features = torch.nn.functional.normalize(
                features, p=2, dim=-1
            )

        embedding = features.squeeze(0).cpu().tolist()

        image_id = (
            f"{DOCUMENT_DIR.name}"
            f"_page_{metadata['page']}"
            f"_image_{metadata['image_number']}"
        )

        collection.upsert(
            ids=[image_id],
            embeddings=[embedding],
            metadatas=[{
                "source": metadata["source"],
                "page": metadata["page"],
                "image_number": metadata["image_number"],
                "image_path": str(image_path),
                "type": "image",
                "model": MODEL_NAME,
            }],
            documents=[str(image_path)],
        )

        indexed += 1

    except Exception as error:
        errors += 1
        print(f"Erreur avec {image_path.name} : {error}")

print("\n--- Indexation des images terminée ---")
print(f"Collection : {collection.name}")
print(f"Images indexées : {indexed}")
print(f"Erreurs : {errors}")
print(f"Vecteurs dans ChromaDB : {collection.count()}")
print(f"Dimension CLIP : {model.config.projection_dim}")
