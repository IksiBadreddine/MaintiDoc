
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MODEL_NAME = "BAAI/bge-m3"

print(f"Chargement du modèle : {MODEL_NAME}")

embed_model = HuggingFaceEmbedding(
    model_name=MODEL_NAME
)

text = (
    "Le compresseur PACIFIC nécessite une vérification "
    "régulière du niveau d'huile."
)

vector = embed_model.get_text_embedding(text)

print("\n--- Test des embeddings ---")
print(f"Dimension du vecteur : {len(vector)}")
print(f"Nombre de valeurs : {len(vector)}")
print(f"Premières valeurs : {vector[:5]}")
print(f"Type du vecteur : {type(vector).__name__}")
